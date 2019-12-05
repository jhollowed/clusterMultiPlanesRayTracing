# this script points to alternative makefiles that explicitly points to gcc-9 
# in the event the MacOS by default has the system gcc pointing to clang, and 
# adds includes and linking flags for fftw3. Also, -lrt is removed from makefiles, 
# and stdlib.h is included in place of malloc.h in lib_so_sph_w_omp/proto.h
# see https://stackoverflow.com/questions/36211018/clang-error-errorunsupported-option-fopenmp-on-mac-osx-el-capitan-buildin/36211162#36211162
echo making lib_so_cgls
cd ./lib_so_cgls/
./make_so_gcc9
cd ..

echo making lib_so_cgls_p
cd ./lib_so_cgls_p/
./make_so_gcc9
cd ..

echo making lib_so_icic
cd ./lib_so_icic/
./make_so_gcc9
cd ..

echo making lib_so_lzos
cd ./lib_so_lzos/
./make_so_gcc9
cd ..

echo making lib_so_lzos
cd ./lib_so_lzos/
./make_so_gcc9
cd ..

echo making lib_so_omp_cic
cd ./lib_so_omp_cic/
./make_so_gcc9
cd ..

echo making lib_so_omp_ngp
cd ./lib_so_omp_ngp/
./make_so_gcc9
cd ..

echo making lib_so_sph_w_omp
cd ./lib_so_sph_w_omp/
./make_so_gcc9
cd ..

echo making lib_so_tri_roots
cd ./lib_so_tri_roots/
./make_so_gcc9
cd ..
