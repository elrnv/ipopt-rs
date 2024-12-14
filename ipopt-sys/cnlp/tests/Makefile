MKLROOT=/home/egor
Ipopt_LIBRARY_DIR=/home/egor/proj/ipopt-rs/ipopt-sys/target/debug/build/ipopt-sys-56a446d7f232a286/out/lib
build:
	gcc hs071_nlp.h hs071_nlp.c hs071_main.c \
	-I../src -L${MKLROOT}/lib \
	${Ipopt_LIBRARY_DIR}/libipopt_cnlp.a \
	${Ipopt_LIBRARY_DIR}/libipopt.a \
	${MKLROOT}/lib/libmkl_intel_lp64.a \
	${MKLROOT}/lib/libmkl_tbb_thread.a \
	${MKLROOT}/lib/libmkl_core.a \
	${MKLROOT}/lib/libmkl_intel_lp64.a \
	${MKLROOT}/lib/libmkl_tbb_thread.a \
	${MKLROOT}/lib/libmkl_core.a \
	${MKLROOT}/lib/libmkl_intel_lp64.a \
	${MKLROOT}/lib/libmkl_tbb_thread.a \
	${MKLROOT}/lib/libmkl_core.a \
	-ltbb \
   	-lstdc++ -lpthread -lm -ldl -fuse-ld=lld

#-Wl,--start-group \
#-Wl,--end-group \
