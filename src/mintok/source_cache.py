"""Source Span Deduplication & Novelty Tracking for MinTok 3.1.

Attacks the 14% source code inference spend bucket.
Eliminates redundant source code re-transmission across exploratory reads by:
1. Tracking exact visible line intervals per file hash (file.py:20-74 @ hash abc).
2. Calculating novelty: Novelty(chunk) = 1 - (already_visible_content / chunk_content).
3. Ranking retrieval chunks by (Relevance * Novelty) / Tokens.
4. Content-addressable source regions: 'src:a83f = foo.py @ hash 726a',
   rendering identical re-reads as 'src:a83f unchanged'.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from mintok.tokens import estimate_tokens


class RegionStatus:
    SEEN = "SEEN"          # Seen previously, but evicted from active model context
    RESIDENT = "RESIDENT"  # Currently resident in compiled model context
    SALIENT = "SALIENT"    # Actively resident AND fresh/high-attention enough to act on


@dataclass(frozen=True, slots=True)
class SourceRegion:
    """An immutable, content-addressable source code span."""

    id: str
    file_path: str
    start_line: int
    end_line: int
    content_hash: str
    tokens: int
    content: str = ""
    status: str = RegionStatus.RESIDENT
    last_referenced_turn: int = 1
    reference_count: int = 1


class SourceCache:
    """Content-addressable cache and interval tracker for visible source spans."""

    def __init__(self) -> None:
        # file_path -> list of (start_line, end_line)
        self._visible_intervals: dict[str, list[tuple[int, int]]] = {}
        self._regions_by_id: dict[str, SourceRegion] = {}
        self._hash_to_id: dict[str, str] = {}

    def compute_novelty(self, file_path: str, start_line: int, end_line: int) -> float:
        """Compute Novelty = 1 - (already_visible_lines / total_chunk_lines)."""
        intervals = self._visible_intervals.get(file_path, [])
        total_lines = max(1, end_line - start_line + 1)
        if not intervals:
            return 1.0

        overlap_lines = 0
        for vs, ve in intervals:
            o_start = max(start_line, vs)
            o_end = min(end_line, ve)
            if o_start <= o_end:
                overlap_lines += (o_end - o_start + 1)

        overlap_lines = min(total_lines, overlap_lines)
        return max(0.0, 1.0 - (overlap_lines / total_lines))

    def mark_evicted(self, src_id: str) -> None:
        """Evict a region from resident prompt context (transitions RESIDENT/SALIENT -> SEEN)."""
        reg = self._regions_by_id.get(src_id)
        if not reg:
            return
        # Replace with SEEN status
        self._regions_by_id[src_id] = SourceRegion(
            id=reg.id,
            file_path=reg.file_path,
            start_line=reg.start_line,
            end_line=reg.end_line,
            content_hash=reg.content_hash,
            tokens=reg.tokens,
            content=reg.content,
            status=RegionStatus.SEEN,
            last_referenced_turn=reg.last_referenced_turn,
            reference_count=reg.reference_count,
        )
        # Remove from active intervals
        if reg.file_path in self._visible_intervals:
            interval = (reg.start_line, reg.end_line)
            if interval in self._visible_intervals[reg.file_path]:
                self._visible_intervals[reg.file_path].remove(interval)

    def mark_salient(self, src_id: str, current_turn: int = 1) -> None:
        """Mark a region as SALIENT (fresh, high attention in context)."""
        reg = self._regions_by_id.get(src_id)
        if not reg:
            return
        self._regions_by_id[src_id] = SourceRegion(
            id=reg.id,
            file_path=reg.file_path,
            start_line=reg.start_line,
            end_line=reg.end_line,
            content_hash=reg.content_hash,
            tokens=reg.tokens,
            content=reg.content,
            status=RegionStatus.SALIENT,
            last_referenced_turn=current_turn,
            reference_count=reg.reference_count + 1,
        )

    def infer_salience(self, src_id: str, current_turn: int = 1, max_salient_age: int = 3) -> str:
        """Infer whether a resident region is SALIENT (fresh) or RESIDENT (low-attention)."""
        reg = self._regions_by_id.get(src_id)
        if not reg:
            return RegionStatus.SEEN
        if reg.status == RegionStatus.SEEN:
            return RegionStatus.SEEN

        age = max(0, current_turn - reg.last_referenced_turn)
        if age <= max_salient_age or reg.reference_count >= 3:
            status = RegionStatus.SALIENT
        else:
            status = RegionStatus.RESIDENT

        if reg.status != status:
            self._regions_by_id[src_id] = SourceRegion(
                id=reg.id,
                file_path=reg.file_path,
                start_line=reg.start_line,
                end_line=reg.end_line,
                content_hash=reg.content_hash,
                tokens=reg.tokens,
                content=reg.content,
                status=status,
                last_referenced_turn=reg.last_referenced_turn,
                reference_count=reg.reference_count,
            )
        return status

    def rehydrate(self, src_id: str, current_turn: int = 1, status: str = RegionStatus.RESIDENT) -> tuple[str, int]:
        """Rehydrate a SEEN region back to RESIDENT/SALIENT status."""
        reg = self._regions_by_id.get(src_id)
        if not reg:
            return f"error: region {src_id} not found", 0
        self._regions_by_id[src_id] = SourceRegion(
            id=reg.id,
            file_path=reg.file_path,
            start_line=reg.start_line,
            end_line=reg.end_line,
            content_hash=reg.content_hash,
            tokens=reg.tokens,
            content=reg.content,
            status=status,
            last_referenced_turn=current_turn,
            reference_count=reg.reference_count + 1,
        )
        if reg.file_path not in self._visible_intervals:
            self._visible_intervals[reg.file_path] = []
        self._visible_intervals[reg.file_path].append((reg.start_line, reg.end_line))

        header = f"[{reg.id} rehydrated: {reg.file_path}:{reg.start_line}-{reg.end_line}]"
        return f"{header}\n{reg.content}", reg.tokens

    def register_span(
        self,
        file_path: str,
        start_line: int,
        end_line: int,
        content: str,
        file_hash: str = "",
    ) -> tuple[str, float]:
        """Register a source span, returning either novel excerpt, handle, or rehydrated text."""
        h = hashlib.sha256(f"{file_path}:{start_line}:{end_line}:{content}".encode("utf-8")).hexdigest()[:8]
        src_id = f"src:{h}"

        if src_id in self._regions_by_id:
            reg = self._regions_by_id[src_id]
            if reg.status in (RegionStatus.RESIDENT, RegionStatus.SALIENT):
                return f"[{src_id} unchanged: {file_path}:{start_line}-{end_line}]", 0.0
            else:
                # Was seen previously, now re-requested -> rehydrate into active context
                text, _ = self.rehydrate(src_id)
                return text, 1.0

        novelty = self.compute_novelty(file_path, start_line, end_line)

        # Record interval
        if file_path not in self._visible_intervals:
            self._visible_intervals[file_path] = []
        self._visible_intervals[file_path].append((start_line, end_line))

        reg = SourceRegion(
            id=src_id,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            content_hash=h,
            tokens=estimate_tokens(content),
            content=content,
            status=RegionStatus.RESIDENT,
        )
        self._regions_by_id[src_id] = reg
        self._hash_to_id[h] = src_id

        header = f"[{src_id}: {file_path}:{start_line}-{end_line}]"
        return f"{header}\n{content}", novelty

    def rank_chunks(
        self,
        chunks: list[dict[str, Any]],
        query: str,
    ) -> list[dict[str, Any]]:
        """Rank chunks by (Relevance * Novelty) / Tokens."""
        query_terms = [q.lower() for q in re.findall(r"\w+", query) if len(q) > 2]
        scored_chunks = []

        for ch in chunks:
            fpath = ch.get("file_path", "")
            s_line = ch.get("start_line", 1)
            e_line = ch.get("end_line", s_line + 10)
            content = ch.get("content", "")
            toks = max(20, ch.get("tokens", estimate_tokens(content)))

            # Term overlap relevance
            c_lower = content.lower()
            relevance = 0.2
            if query_terms:
                matches = sum(1 for term in query_terms if term in c_lower)
                relevance += 0.8 * (matches / len(query_terms))

            novelty = self.compute_novelty(fpath, s_line, e_line)
            priority = (relevance * (0.2 + 0.8 * novelty)) / float(toks)

            scored_ch = dict(ch)
            scored_ch["relevance"] = round(relevance, 3)
            scored_ch["novelty"] = round(novelty, 3)
            scored_ch["priority"] = round(priority, 6)
            scored_chunks.append(scored_ch)

        scored_chunks.sort(key=lambda x: x["priority"], reverse=True)
        return scored_chunks
