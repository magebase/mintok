@domain
Feature: Import resolution through package re-exports
  Call edges must follow names re-exported through a package ``__init__``, or
  the compiler silently misses callers and slices.

  Scenario: A name re-exported from a package resolves to its defining module
    Given a repository file "pkg/__init__.py":
      """
      from pkg.impl import Thing
      """
    And a repository file "pkg/impl.py":
      """
      class Thing:
          def spin(self):
              return True
      """
    And a repository file "app.py":
      """
      from pkg import Thing


      def run(thing: Thing):
          return thing.spin()
      """
    When the repository is compiled
    Then symbol "app:run" calls "pkg.impl:Thing.spin" with confidence 1.00

  Scenario: A function re-exported from a package resolves to its defining module
    Given a repository file "helpers/__init__.py":
      """
      from helpers.stringutil import slugify
      """
    And a repository file "helpers/stringutil.py":
      """
      def slugify(value):
          return value.strip().lower().replace(" ", "-")
      """
    And a repository file "blog.py":
      """
      from helpers import slugify


      def make_slug(title):
          return slugify(title)
      """
    When the repository is compiled
    Then symbol "blog:make_slug" calls "helpers.stringutil:slugify" with confidence 1.00
