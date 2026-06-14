Julia OS X packaging
====================

This builds the Julia OS X application bundle (.app folder), and stores it in a disk image
(.dmg file).

The application bundle is a small launcher executable which opens Terminal.app and runs the
julia binary (which opens the REPL). All the Julia binary files and their dependencies are
bundled inside this.

Run `make` to build.

Other files in this directory

* `launcher.c` is the launcher source, compiled to the bundle's executable.
* `Info.plist` is the bundle metadata template (version fields are filled in at build time).
* `julia.icns` is the Julia icon file.
