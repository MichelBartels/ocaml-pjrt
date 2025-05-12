(lang dune 3.16)

(using ctypes 0.3)

(name pjrt_bindings)

(generate_opam_files true)

(source
 (github MichelBartels/ocaml-pjrt))

(authors "Michel Bartels")

(maintainers "Michel Bartels")

(license LICENSE)

(package
 (name pjrt_bindings)
 (depends ocaml dune ctypes ctypes-foreign device_api))
