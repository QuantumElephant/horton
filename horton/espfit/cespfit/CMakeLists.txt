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
  ${CMAKE_CURRENT_SOURCE_DIR}/electrostatics.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mask.cpp
)

# Define header files
set(HEADER_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/electrostatics.h
  ${CMAKE_CURRENT_SOURCE_DIR}/mask.h
)

# Define the shared library
add_library(cespfit SHARED ${SOURCE_FILES})
set_property(TARGET cespfit PROPERTY VERSION ${ESPFIT_VERSION})
set_property(TARGET cespfit PROPERTY SOVERSION ${ESPFIT_SOVERSION})

# Set external libraries for linking
set(CYTHON_LINK ${PYTHON_LIBRARIES} cespfit PARENT_SCOPE)

# Install include files
install (FILES ${HEADER_FILES} DESTINATION include/cespfit)

# Install library
install (TARGETS cespfit DESTINATION lib)

# Unit Tests with Google testing framework
# add_subdirectory(tests)
