CC = gcc
CFLAGS = -O0 -g

int-lenet: int-lenet.o int8_t_images.o
float-lenet: float-lenet.o float_images.o

int8_t_images.c : dump-images.py
float_images.c : dump-images.py

int8_t_parameters.c : dump-parameters.py
float_parameters.c : dump-parameters.py
	
clean : 
	$(RM) ilenet flenet *.o
