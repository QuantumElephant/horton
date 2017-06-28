# If the pyx file is a C++ file, we should specify that here.
set_source_files_properties( cext.pyx PROPERTIES CYTHON_IS_CXX TRUE )

# Multi-file cython modules do not appear to be working at the moment.
cython_add_module( cext cext.pyx )
