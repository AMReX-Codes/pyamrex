.. _developers-release:

Dependencies & Releases
=======================

Update pyAMReX' Core Dependencies
---------------------------------

pyAMReX has a direct dependency on AMReX, which we periodically update.

It further depends on pybind11 and a Python interpreter.


Create a new pyAMReX release
----------------------------

pyAMReX has one release per month.
The version number is set at the beginning of the month and follows the format ``YY.MM``.

In order to create a GitHub release, you need to:

 1. Create a new branch from ``development`` and update the version number in all source files.
    We usually wait for the AMReX release to be tagged first, then we also point to its tag.

    For a pyAMReX release, ideally a *git tag* of AMReX shall be used instead of an unnamed commit.

    Then open a PR, wait for tests to pass and then merge.

 2. **Local Commit** (Optional): at the moment, ``@ax3l`` is managing releases and signs tags (naming: ``YY.MM``) locally with his GPG key before uploading them to GitHub.

    **Publish**: On the `GitHub Release page <https://github.com/AMReX-Codes/pyamrex/releases>`__, create a new release via ``Draft a new release``.
    Either select the locally created tag or create one online (naming: ``YY.MM``) on the merged commit of the PR from step 1.

    In the *release description*, please specify the compatible versions of dependencies (see previous releases), and provide info on the content of the release.
    In order to get a list of PRs merged since last release, you may run

    .. code-block:: sh

       git log <last-release-tag>.. --format='- %s'

 3. Optional/future: create a ``release-<version>`` branch, write a changelog, and backport bug-fixes for a few days.
