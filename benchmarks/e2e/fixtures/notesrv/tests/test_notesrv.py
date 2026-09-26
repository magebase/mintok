import json

import pytest

from notesrv.handlers import Handlers
from notesrv.jsonutil import dumps, exception_to_response, loads
from notesrv.models import Note, NoteError
from notesrv.routing import RouteTable
from notesrv.server import build_routes, dispatch
from notesrv.storage import MemoryStore, NotFoundError

ADMIN = "tok-admin-001"
READER = "tok-reader-003"


def test_note_model_round_trip():
    note = Note(title="Standup", body="Agenda", tags=[" Work ", "team"])
    stored = Note.from_dict(note.to_dict())
    assert stored == note
    assert note.tags == ["work", "team"]


def test_note_title_required():
    with pytest.raises(NoteError):
        Note(title="   ", body="")


def test_store_crud():
    store = MemoryStore()
    note = store.create(Note(title="One", body=""))
    store.create(Note(title="Two", body=""))
    assert store.get(2).title == "Two"
    assert [n.title for n in store.list()] == ["One", "Two"]
    store.update(1, {"body": "updated"})
    assert store.get(1).body == "updated"
    assert store.delete(1) is True
    assert store.delete(1) is False
    with pytest.raises(NotFoundError):
        store.get(1)


def test_store_tag_filter_and_search():
    store = MemoryStore()
    store.create(Note(title="Groceries", body="milk", tags=["personal"]))
    store.create(Note(title="Retro", body="sprint", tags=["work", "team"]))
    assert len(store.list(tag="work")) == 1
    assert store.search("SPRINT")[0].title == "Retro"


def test_handlers_auth_enforced():
    handlers = Handlers()
    status, payload = handlers.handle_create(ADMIN, {"title": "Secret"})
    assert status == 201
    status, _ = handlers.handle_list(READER)
    assert status == 200
    status, payload = handlers.handle_delete(READER, 1)
    assert status == 403
    status, payload = handlers.handle_get(None, 1)
    assert status == 403


def test_handlers_not_found():
    handlers = Handlers()
    status, payload = handlers.handle_get(ADMIN, 99)
    assert status == 404
    assert "error" in payload


def test_routing_resolution():
    table = RouteTable()
    table.add_route("GET", "/notes", lambda **kw: (200, {}))
    table.add_route("GET", "/notes/<note_id>", lambda **kw: (200, {}))
    resolved = table.resolve("get", "/notes/42")
    assert resolved is not None and resolved.params == {"note_id": "42"}
    assert table.resolve("GET", "/other") is None
    assert table.resolve("POST", "/notes/42") is None
    with pytest.raises(ValueError):
        table.add_route("GET", "/notes", lambda **kw: (200, {}))


def test_dispatch_full_flow():
    handlers = Handlers()
    routes = build_routes(handlers)
    status, text = dispatch(routes, handlers, "POST", "/notes", json.dumps({"title": "N1"}), token=ADMIN)
    assert status == 201 and loads(text)["title"] == "N1"
    status, text = dispatch(routes, handlers, "GET", "/notes", "", token=READER)
    assert status == 200 and len(loads(text)["notes"]) == 1
    status, text = dispatch(routes, handlers, "GET", "/notes/1", "", token=READER)
    assert status == 200
    status, text = dispatch(routes, handlers, "PUT", "/notes/1", json.dumps({"body": "b2"}), token=ADMIN)
    assert status == 200 and loads(text)["body"] == "b2"
    status, text = dispatch(routes, handlers, "DELETE", "/notes/1", "", token=ADMIN)
    assert status == 204
    status, text = dispatch(routes, handlers, "GET", "/notes/1", "", token=READER)
    assert status == 404
    status, text = dispatch(routes, handlers, "GET", "/nope", "")
    assert status == 404 and "no route" in text


def test_dispatch_bad_note_id_and_body():
    handlers = Handlers()
    routes = build_routes(handlers)
    status, text = dispatch(routes, handlers, "GET", "/notes/abc", "")
    assert status == 400
    status, text = dispatch(routes, handlers, "POST", "/notes", "not json")
    assert status == 400


def test_dumps_is_sorted_and_compact():
    text = dumps({"b": 1, "a": 2})
    assert text == '{"a":2,"b":1}'
