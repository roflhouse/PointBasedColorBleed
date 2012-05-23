#  CPE 473
#  -------------------
#  @author Nick Feeney
CC=g++
LD=g++
CFLAGS= -Wall  -I "./glm" -g -c -O3 
LDFLAGS= -Wall -I "./glm" -g -O3

ALL= PBCMain.o Util/Header.o Util/Tga.o Objects/Sphere.o Objects/LightSource.o Objects/Plane.o Objects/ObjectInfo.o Objects/Camera.o Util/Parser.o Objects/Triangle.o

all:	$(ALL)  raytracer

raytracer:	$(ALL)
	$(CC) $(LDFLAGS) $(ALL) -o raytracer 

raytracer.o:	raytracer.cpp Util/Header.h Util/Tga.h Objects/Sphere.h Objects/Objects.h Objects/Plane.h Util/vec3.h Objects/LightSource.h Objects/Camera.h Util/Parser.h 
	$(CC) $(CFLAGS) -o $@ $<

Util/Header.o:	Util/Header.cpp Util/Header.h
	$(CC) $(CFLAGS) -o $@ $<

Util/Tga.o:	Util/Tga.cpp Util/Tga.h Util/Header.h
	$(CC) $(CFLAGS) -o $@ $<

Objects/Sphere.o:	Objects/Sphere.cpp Objects/Sphere.h Objects/Objects.h Util/vec3.h
	$(CC) $(CFLAGS) -o $@ $<

Objects/Triangle.o:	Objects/Triangle.cpp Objects/Triangle.h Objects/Objects.h Util/vec3.h
	$(CC) $(CFLAGS) -o $@ $<

Objects/Plane.o:	Objects/Plane.cpp Objects/Plane.h Objects/Objects.h Util/vec3.h
	$(CC) $(CFLAGS) -o $@ $<

Objects/LightSource.o:	Objects/LightSource.cpp Objects/LightSource.h Util/vec3.h 
	$(CC) $(CFLAGS) -o $@ $<

Objects/Camera.o:	Objects/Camera.cpp Objects/Camera.h Util/vec3.h Util/Ray.h 
	$(CC) $(CFLAGS) -o $@ $<

Objects/Object.o:	Objects/Object.cpp Objects/Objects.h Util/vec3.h  
	$(CC) $(CFLAGS) -o $@ $<

Util/Parser.o:	Util/Parser.cpp Util/Parser.h Objects/Objects.h Objects/Sphere.h Objects/Plane.h Objects/Camera.h Objects/LightSource.h 
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -rf core* *.o *.gch $(ALL) junk*
