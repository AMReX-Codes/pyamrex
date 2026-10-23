# CDash dashboard settings, read by include(CTest) in set_testing_option
# (cmake/pyAMReXFunctions.cmake).
#
# Dashboard: https://my.cdash.org/index.php?project=pyAMReX
#
# Site and build names are set per CI job at configure time via the CTest
# module variables SITE and BUILDNAME, e.g.,
#   cmake -S . -B build -DSITE=GitHub -DBUILDNAME=Linux-GCC-DP-MPI-Debug

set(CTEST_PROJECT_NAME pyAMReX)
set(CTEST_NIGHTLY_START_TIME 08:00:00 UTC)

set(CTEST_SUBMIT_URL https://my.cdash.org/submit.php?project=pyAMReX)

set(CTEST_DROP_SITE_CDASH TRUE)
