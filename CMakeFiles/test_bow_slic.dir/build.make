# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.4

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /Applications/CMake.app/Contents/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/NikithaShravan/Downloads/BOW-Slic_EDGE

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/NikithaShravan/Downloads/BOW-Slic_EDGE

# Include any dependencies generated for this target.
include CMakeFiles/test_bow_slic.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/test_bow_slic.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_bow_slic.dir/flags.make

CMakeFiles/test_bow_slic.dir/src/test_bow_slic.cpp.o: CMakeFiles/test_bow_slic.dir/flags.make
CMakeFiles/test_bow_slic.dir/src/test_bow_slic.cpp.o: src/test_bow_slic.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/NikithaShravan/Downloads/BOW-Slic_EDGE/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test_bow_slic.dir/src/test_bow_slic.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_bow_slic.dir/src/test_bow_slic.cpp.o -c /Users/NikithaShravan/Downloads/BOW-Slic_EDGE/src/test_bow_slic.cpp

CMakeFiles/test_bow_slic.dir/src/test_bow_slic.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_bow_slic.dir/src/test_bow_slic.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/NikithaShravan/Downloads/BOW-Slic_EDGE/src/test_bow_slic.cpp > CMakeFiles/test_bow_slic.dir/src/test_bow_slic.cpp.i

CMakeFiles/test_bow_slic.dir/src/test_bow_slic.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_bow_slic.dir/src/test_bow_slic.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/NikithaShravan/Downloads/BOW-Slic_EDGE/src/test_bow_slic.cpp -o CMakeFiles/test_bow_slic.dir/src/test_bow_slic.cpp.s

CMakeFiles/test_bow_slic.dir/src/test_bow_slic.cpp.o.requires:

.PHONY : CMakeFiles/test_bow_slic.dir/src/test_bow_slic.cpp.o.requires

CMakeFiles/test_bow_slic.dir/src/test_bow_slic.cpp.o.provides: CMakeFiles/test_bow_slic.dir/src/test_bow_slic.cpp.o.requires
	$(MAKE) -f CMakeFiles/test_bow_slic.dir/build.make CMakeFiles/test_bow_slic.dir/src/test_bow_slic.cpp.o.provides.build
.PHONY : CMakeFiles/test_bow_slic.dir/src/test_bow_slic.cpp.o.provides

CMakeFiles/test_bow_slic.dir/src/test_bow_slic.cpp.o.provides.build: CMakeFiles/test_bow_slic.dir/src/test_bow_slic.cpp.o


CMakeFiles/test_bow_slic.dir/src/bow_slic.cpp.o: CMakeFiles/test_bow_slic.dir/flags.make
CMakeFiles/test_bow_slic.dir/src/bow_slic.cpp.o: src/bow_slic.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/NikithaShravan/Downloads/BOW-Slic_EDGE/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/test_bow_slic.dir/src/bow_slic.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_bow_slic.dir/src/bow_slic.cpp.o -c /Users/NikithaShravan/Downloads/BOW-Slic_EDGE/src/bow_slic.cpp

CMakeFiles/test_bow_slic.dir/src/bow_slic.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_bow_slic.dir/src/bow_slic.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/NikithaShravan/Downloads/BOW-Slic_EDGE/src/bow_slic.cpp > CMakeFiles/test_bow_slic.dir/src/bow_slic.cpp.i

CMakeFiles/test_bow_slic.dir/src/bow_slic.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_bow_slic.dir/src/bow_slic.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/NikithaShravan/Downloads/BOW-Slic_EDGE/src/bow_slic.cpp -o CMakeFiles/test_bow_slic.dir/src/bow_slic.cpp.s

CMakeFiles/test_bow_slic.dir/src/bow_slic.cpp.o.requires:

.PHONY : CMakeFiles/test_bow_slic.dir/src/bow_slic.cpp.o.requires

CMakeFiles/test_bow_slic.dir/src/bow_slic.cpp.o.provides: CMakeFiles/test_bow_slic.dir/src/bow_slic.cpp.o.requires
	$(MAKE) -f CMakeFiles/test_bow_slic.dir/build.make CMakeFiles/test_bow_slic.dir/src/bow_slic.cpp.o.provides.build
.PHONY : CMakeFiles/test_bow_slic.dir/src/bow_slic.cpp.o.provides

CMakeFiles/test_bow_slic.dir/src/bow_slic.cpp.o.provides.build: CMakeFiles/test_bow_slic.dir/src/bow_slic.cpp.o


CMakeFiles/test_bow_slic.dir/src/oversegmentation.cpp.o: CMakeFiles/test_bow_slic.dir/flags.make
CMakeFiles/test_bow_slic.dir/src/oversegmentation.cpp.o: src/oversegmentation.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/NikithaShravan/Downloads/BOW-Slic_EDGE/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/test_bow_slic.dir/src/oversegmentation.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_bow_slic.dir/src/oversegmentation.cpp.o -c /Users/NikithaShravan/Downloads/BOW-Slic_EDGE/src/oversegmentation.cpp

CMakeFiles/test_bow_slic.dir/src/oversegmentation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_bow_slic.dir/src/oversegmentation.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/NikithaShravan/Downloads/BOW-Slic_EDGE/src/oversegmentation.cpp > CMakeFiles/test_bow_slic.dir/src/oversegmentation.cpp.i

CMakeFiles/test_bow_slic.dir/src/oversegmentation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_bow_slic.dir/src/oversegmentation.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/NikithaShravan/Downloads/BOW-Slic_EDGE/src/oversegmentation.cpp -o CMakeFiles/test_bow_slic.dir/src/oversegmentation.cpp.s

CMakeFiles/test_bow_slic.dir/src/oversegmentation.cpp.o.requires:

.PHONY : CMakeFiles/test_bow_slic.dir/src/oversegmentation.cpp.o.requires

CMakeFiles/test_bow_slic.dir/src/oversegmentation.cpp.o.provides: CMakeFiles/test_bow_slic.dir/src/oversegmentation.cpp.o.requires
	$(MAKE) -f CMakeFiles/test_bow_slic.dir/build.make CMakeFiles/test_bow_slic.dir/src/oversegmentation.cpp.o.provides.build
.PHONY : CMakeFiles/test_bow_slic.dir/src/oversegmentation.cpp.o.provides

CMakeFiles/test_bow_slic.dir/src/oversegmentation.cpp.o.provides.build: CMakeFiles/test_bow_slic.dir/src/oversegmentation.cpp.o


CMakeFiles/test_bow_slic.dir/src/debugging_functions.cpp.o: CMakeFiles/test_bow_slic.dir/flags.make
CMakeFiles/test_bow_slic.dir/src/debugging_functions.cpp.o: src/debugging_functions.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/NikithaShravan/Downloads/BOW-Slic_EDGE/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/test_bow_slic.dir/src/debugging_functions.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_bow_slic.dir/src/debugging_functions.cpp.o -c /Users/NikithaShravan/Downloads/BOW-Slic_EDGE/src/debugging_functions.cpp

CMakeFiles/test_bow_slic.dir/src/debugging_functions.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_bow_slic.dir/src/debugging_functions.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/NikithaShravan/Downloads/BOW-Slic_EDGE/src/debugging_functions.cpp > CMakeFiles/test_bow_slic.dir/src/debugging_functions.cpp.i

CMakeFiles/test_bow_slic.dir/src/debugging_functions.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_bow_slic.dir/src/debugging_functions.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/NikithaShravan/Downloads/BOW-Slic_EDGE/src/debugging_functions.cpp -o CMakeFiles/test_bow_slic.dir/src/debugging_functions.cpp.s

CMakeFiles/test_bow_slic.dir/src/debugging_functions.cpp.o.requires:

.PHONY : CMakeFiles/test_bow_slic.dir/src/debugging_functions.cpp.o.requires

CMakeFiles/test_bow_slic.dir/src/debugging_functions.cpp.o.provides: CMakeFiles/test_bow_slic.dir/src/debugging_functions.cpp.o.requires
	$(MAKE) -f CMakeFiles/test_bow_slic.dir/build.make CMakeFiles/test_bow_slic.dir/src/debugging_functions.cpp.o.provides.build
.PHONY : CMakeFiles/test_bow_slic.dir/src/debugging_functions.cpp.o.provides

CMakeFiles/test_bow_slic.dir/src/debugging_functions.cpp.o.provides.build: CMakeFiles/test_bow_slic.dir/src/debugging_functions.cpp.o


# Object files for target test_bow_slic
test_bow_slic_OBJECTS = \
"CMakeFiles/test_bow_slic.dir/src/test_bow_slic.cpp.o" \
"CMakeFiles/test_bow_slic.dir/src/bow_slic.cpp.o" \
"CMakeFiles/test_bow_slic.dir/src/oversegmentation.cpp.o" \
"CMakeFiles/test_bow_slic.dir/src/debugging_functions.cpp.o"

# External object files for target test_bow_slic
test_bow_slic_EXTERNAL_OBJECTS =

test_bow_slic: CMakeFiles/test_bow_slic.dir/src/test_bow_slic.cpp.o
test_bow_slic: CMakeFiles/test_bow_slic.dir/src/bow_slic.cpp.o
test_bow_slic: CMakeFiles/test_bow_slic.dir/src/oversegmentation.cpp.o
test_bow_slic: CMakeFiles/test_bow_slic.dir/src/debugging_functions.cpp.o
test_bow_slic: CMakeFiles/test_bow_slic.dir/build.make
test_bow_slic: /usr/local/lib/libopencv_videostab.3.1.0.dylib
test_bow_slic: /usr/local/lib/libopencv_superres.3.1.0.dylib
test_bow_slic: /usr/local/lib/libopencv_stitching.3.1.0.dylib
test_bow_slic: /usr/local/lib/libopencv_shape.3.1.0.dylib
test_bow_slic: /usr/local/lib/libopencv_photo.3.1.0.dylib
test_bow_slic: /usr/local/lib/libopencv_objdetect.3.1.0.dylib
test_bow_slic: /usr/local/lib/libopencv_calib3d.3.1.0.dylib
test_bow_slic: /usr/local/lib/libopencv_features2d.3.1.0.dylib
test_bow_slic: /usr/local/lib/libopencv_ml.3.1.0.dylib
test_bow_slic: /usr/local/lib/libopencv_highgui.3.1.0.dylib
test_bow_slic: /usr/local/lib/libopencv_videoio.3.1.0.dylib
test_bow_slic: /usr/local/lib/libopencv_imgcodecs.3.1.0.dylib
test_bow_slic: /usr/local/lib/libopencv_flann.3.1.0.dylib
test_bow_slic: /usr/local/lib/libopencv_video.3.1.0.dylib
test_bow_slic: /usr/local/lib/libopencv_imgproc.3.1.0.dylib
test_bow_slic: /usr/local/lib/libopencv_core.3.1.0.dylib
test_bow_slic: CMakeFiles/test_bow_slic.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/NikithaShravan/Downloads/BOW-Slic_EDGE/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable test_bow_slic"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_bow_slic.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_bow_slic.dir/build: test_bow_slic

.PHONY : CMakeFiles/test_bow_slic.dir/build

CMakeFiles/test_bow_slic.dir/requires: CMakeFiles/test_bow_slic.dir/src/test_bow_slic.cpp.o.requires
CMakeFiles/test_bow_slic.dir/requires: CMakeFiles/test_bow_slic.dir/src/bow_slic.cpp.o.requires
CMakeFiles/test_bow_slic.dir/requires: CMakeFiles/test_bow_slic.dir/src/oversegmentation.cpp.o.requires
CMakeFiles/test_bow_slic.dir/requires: CMakeFiles/test_bow_slic.dir/src/debugging_functions.cpp.o.requires

.PHONY : CMakeFiles/test_bow_slic.dir/requires

CMakeFiles/test_bow_slic.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_bow_slic.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_bow_slic.dir/clean

CMakeFiles/test_bow_slic.dir/depend:
	cd /Users/NikithaShravan/Downloads/BOW-Slic_EDGE && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/NikithaShravan/Downloads/BOW-Slic_EDGE /Users/NikithaShravan/Downloads/BOW-Slic_EDGE /Users/NikithaShravan/Downloads/BOW-Slic_EDGE /Users/NikithaShravan/Downloads/BOW-Slic_EDGE /Users/NikithaShravan/Downloads/BOW-Slic_EDGE/CMakeFiles/test_bow_slic.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_bow_slic.dir/depend

