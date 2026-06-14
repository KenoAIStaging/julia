// Julia.app launcher: open a Terminal window running the bundled `julia`.
//
// This is the compiled replacement for the old AppleScript applet
// (startup.applescript). It reproduces exactly what that applet did --
//
//     open -a Terminal "<Julia.app>/Contents/Resources/julia/bin/julia"
//
// -- but as a tiny native executable, so building Julia.app no longer needs
// `osacompile` (and therefore no longer needs a Mac in the packaging step).
//
// It is the bundle's CFBundleExecutable: double-clicking Julia.app runs this,
// which execs /usr/bin/open to launch Terminal on the bundled julia binary.

#include <limits.h>
#include <mach-o/dyld.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    // Path to this executable: <Julia.app>/Contents/MacOS/<name>
    char raw[PATH_MAX];
    uint32_t size = sizeof(raw);
    if (_NSGetExecutablePath(raw, &size) != 0) {
        fprintf(stderr, "julia-launcher: executable path too long\n");
        return 1;
    }
    char path[PATH_MAX];
    if (realpath(raw, path) == NULL) {
        perror("julia-launcher: realpath");
        return 1;
    }

    // Strip "/<name>" then "/MacOS" to reach <Julia.app>/Contents.
    for (int i = 0; i < 2; i++) {
        char *slash = strrchr(path, '/');
        if (slash == NULL) {
            fprintf(stderr, "julia-launcher: unexpected bundle layout\n");
            return 1;
        }
        *slash = '\0';
    }

    char julia[PATH_MAX];
    if ((size_t)snprintf(julia, sizeof(julia), "%s/Resources/julia/bin/julia",
                         path) >= sizeof(julia)) {
        fprintf(stderr, "julia-launcher: path too long\n");
        return 1;
    }

    execl("/usr/bin/open", "open", "-a", "Terminal", julia, (char *)NULL);
    perror("julia-launcher: exec /usr/bin/open");
    return 1;
}
