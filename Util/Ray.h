/**
 *  CPE 2010
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */

#ifndef RAY_H
#define RAY_H
#include "../Objects/Object.h"
#include <vector>
#include "Vector.h"
#include "../Objects/LightSource.h"
#include "BVH.h"

#include "../Cuda/CudaDefs.h"
#include "../Cuda/CudaSwitch.h"

#define MAX_DEPTH 6
#define MONTE_CARLO_DEPTH 5
#define MONTE_CARLO_RAY 0

extern Object **spheres;
extern Object **planes;
extern Object **triangles;
extern int numPlanes;
extern int numTriangles;
extern int numSpheres;
extern LightSource **lights;
extern BVH *bvh;

class Ray
{
    public:
        Ray( Vector startPos, Vector eyePos, int pixelW, int pixelH );
        Ray( Vector startPos, Vector eyePos, int pixelW, int pixelH, float mod, int depth );
        Ray( Vector startPos, Vector eyePos, int pixelW, int pixelH, float mod, int depth,
                float refractionCur );
        Object::pixel castRay( );
        Object::pixel castRay( float t, int index, float tri_it, int triIndex);
        cuda_ray_t getCudaRay();
        int w;
        int h;
    protected:
        Vector direction;
        Vector position;
        float curDistance;
        bool hit;
        int depth;
        float refractionCur;
};
extern std::vector<Ray *> rays;
#endif
