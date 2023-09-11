# https://discourse.cmake.org/t/how-to-configure-cmake-on-windows-to-use-clang-cl-with-ninja-multi-config-for-intel-32-bits-intel-64-bits-and-arm64-coming-from-visual-studio/3430/10
# https://gitlab.kitware.com/cmake/cmake/-/blob/master/.gitlab/ci/vcvarsall.ps1

$erroractionpreference = "stop"

cmd /c "`"$env:VCVARSALL`" $env:VCVARSPLATFORM -vcvars_ver=$env:VCVARSVERSION & set" |
foreach {
    if ($_ -match "=") {
        $v = $_.split("=")
        [Environment]::SetEnvironmentVariable($v[0], $v[1])
    }
}
