#!/bin/bash
rm -f labels
grep label float_images.c|cut -d' ' -f 3 >> labels
#for fptype in float double fp16 bfloat16 fp8e4m3 fp8e5m2 ; do
for fptype in fp8e5m2 ; do
	touch ./floatx-lenet.cpp
	rm -f result-$fptype accuracy-$fptype
	make DEFFP=$fptype && mv ./floatx-lenet ./floatx-lenet-$fptype
#	{
#	for i in {0..9999} ; do
#		./floatx-lenet-$fptype $i | cut -d' ' -f 3 >> result-$fptype
#	done 
#	diff -U 0 labels result-$fptype | grep -c ^@ >> accuracy-$fptype
#	} &
	for i in {2..255} ; do
		./floatx-lenet-$fptype $i >> result-$fptype-$i &
	done
done
