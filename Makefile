#  CPE 473
#  -------------------
#  @author Nick Feeney
CC=nvcc
LD=nvcc
CFLAGS=  -c -O3 
CFLAGSCUDA= -c -O3 -arch=sm_21 
LDFLAGS= -g -O3 

ALL= Util/Header.o Util/Tga.o Objects/Sphere.o Objects/LightSource.o Objects/Plane.o Objects/ObjectInfo.o Objects/Camera.o Util/Parser.o Objects/Triangle.o Util/vec3.o Util/Ray.o Util/Intersection.o Objects/Surfel.o Util/Octree.o Util/BoundingBox.o  Util/CudaOctree.o Util/CudaRay.o

all:	$(ALL) PBC 

PBC:	$(ALL) PBCMain.o
	$(CC) $(LDFLAGS) PBCMain.o $(ALL) -o PBC 

PBCTest: $(ALL) test.o
	$(CC) $(LDFLAGS) test.o $(ALL) -o PBCTest 
   
test.o:	test.cpp Util/Header.h Util/Tga.h Objects/Sphere.h Objects/Objects.h Objects/Plane.h Util/vec3.h Objects/LightSource.h Objects/Camera.h Util/Parser.h Util/Ray.o  
	$(CC) $(CFLAGS) -o $@ $<

PBCMain.o:	PBCMain.cpp Util/Header.h Util/Tga.h Objects/Sphere.h Objects/Objects.h Objects/Plane.h Util/vec3.h Objects/LightSource.h Objects/Camera.h Util/Parser.h  Util/Ray.o  
	$(CC) $(CFLAGS) -o $@ $<

Util/Header.o:	Util/Header.cpp Util/Header.h
	$(CC) $(CFLAGS) -o $@ $<

Util/BoundingBox.o:	Util/BoundingBox.cpp Util/BoundingBox.h Util/vec3.h
	$(CC) $(CFLAGS) -o $@ $<

Util/CudaRay.o:	Util/CudaRay.cu Util/vec3.h Objects/Surfel.h
	$(CC) $(CFLAGSCUDA) -o $@ $<
   
Util/CudaOctree.o:	Util/CudaOctree.cu  Util/vec3.h Objects/SurfelType.h Util/OctreeType.h
	$(CC) $(CFLAGSCUDA) -o $@ $<

Util/Octree.o:	Util/Octree.cpp Objects/Surfel.h Util/vec3.h
	$(CC) $(CFLAGS) -o $@ $<

Util/Intersection.o:	Util/Intersection.cpp Util/Intersection.h Util/ColorType.h Util/vec3.h
	$(CC) $(CFLAGS) -o $@ $<

Util/vec3.o:	Util/vec3.cpp Util/vec3.h
	$(CC) $(CFLAGS) -o $@ $<

Util/Tga.o:	Util/Tga.cpp Util/Tga.h Util/Header.h
	$(CC) $(CFLAGS) -o $@ $<

Objects/Sphere.o:	Objects/Sphere.cpp Objects/Sphere.h Objects/Objects.h Util/vec3.h
	$(CC) $(CFLAGS) -o $@ $<

Objects/Triangle.o:	Objects/Triangle.cpp Objects/Triangle.h Objects/Objects.h Util/vec3.h
	$(CC) $(CFLAGS) -o $@ $<

Objects/Plane.o:	Objects/Plane.cpp Objects/Plane.h Objects/Objects.h Util/vec3.h
	$(CC) $(CFLAGS) -o $@ $<

Objects/Surfel.o:	Objects/Surfel.cpp Objects/Surfel.h Objects/Objects.h Util/vec3.h Util/Ray.h
	$(CC) $(CFLAGS) -o $@ $<

Objects/LightSource.o:	Objects/LightSource.cpp Objects/LightSource.h Util/vec3.h 
	$(CC) $(CFLAGS) -o $@ $<

Objects/Camera.o:	Objects/Camera.cpp Objects/Camera.h Util/vec3.h Util/Ray.h 
	$(CC) $(CFLAGS) -o $@ $<

Objects/ObjectInfo.o:	Objects/ObjectInfo.cpp Objects/ObjectInfo.h Util/vec3.h  
	$(CC) $(CFLAGS) -o $@ $<

Util/Parser.o:	Util/Parser.cpp Util/Parser.h Objects/Objects.h Objects/Sphere.h Objects/Plane.h Objects/Camera.h Objects/LightSource.h Util/vec3.h Util/ColorType.h 
	$(CC) $(CFLAGS) -o $@ $<

Util/Ray.o:	Util/Ray.cpp Util/Ray.h Objects/Objects.h Objects/Sphere.h Objects/Plane.h Objects/Camera.h Objects/LightSource.h Util/vec3.h Util/ColorType.h Util/BoundingBox.h
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -rf core* *.o *.gch $(ALL) junk*
