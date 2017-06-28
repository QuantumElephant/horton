# CellList is a 3D domain decomposition library.
# Copyright (C) 2011-2015 The CellList Development Team
#
# This file is part of CellList.
#
# CellList is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# CellList is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
#--

# Configure a header file to pass some of the CMake settings
# to the source code.
configure_file (
  ${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/config.h
)

# Explain that the current directory has to be in the source path/
# This allows use to use include "celllists/foo.h", which is consistent
# with the way external programs would compile against celllists.
include_directories(${PROJECT_SOURCE_DIR})

# Define source files
set(SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/boys.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/boys_inc.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/calc.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cartpure.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cholesky.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/common.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fns.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gbasis.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gbw.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ints.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iter_gb.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iter_pow.cpp
)

# Define header files
set(HEADER_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/boys.h
  ${CMAKE_CURRENT_SOURCE_DIR}/calc.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cartpure.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cholesky.h
  ${CMAKE_CURRENT_SOURCE_DIR}/common.h
  ${CMAKE_CURRENT_SOURCE_DIR}/fns.h
  ${CMAKE_CURRENT_SOURCE_DIR}/gbasis.h
  ${CMAKE_CURRENT_SOURCE_DIR}/gbw.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ints.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iter_gb.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iter_pow.h
)

# Define the shared library
add_library(cgbasis SHARED ${SOURCE_FILES})
set_property(TARGET cgbasis PROPERTY VERSION ${CGBASIS_VERSION})
set_property(TARGET cgbasis PROPERTY SOVERSION ${CGBASIS_SOVERSION})

# Set external libraries for linking
set(CYTHON_LINK ${PYTHON_LIBRARIES} cgbasis PARENT_SCOPE)

# Install include files
install(FILES ${HEADER_FILES} DESTINATION include/cgbasis)

# Install library
install (TARGETS cgbasis DESTINATION lib)

# Unit Tests with Google testing framework
# add_subdirectory(tests)
