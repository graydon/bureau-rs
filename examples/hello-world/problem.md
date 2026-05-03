# Problem: A small "echo" CLI

Build a Rust crate that produces a single binary `echo-it` which:

- Reads command-line arguments (anything after the program name)
- Joins them with a single space
- Prints the result to stdout, followed by a newline

If no arguments are given, the program should print a single newline and exit
successfully (status 0).

The crate should be small (3-5 source files) and have at least one integration
test that runs the binary as a subprocess and checks stdout.
