/**
 *  CPE 2010
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */
#include "Sphere.h"

Sphere::Sphere()
{
}
Object::rayInfo Sphere::rayIntersect( Vector direction, Vector position, float t0 )
{
    direction = direction.unit();
    Object::rayInfo ret;
    glm::vec4 dir = glm::vec4(direction.x, direction.y, direction.z, 0.0f);
    glm::vec4 pos = glm::vec4(position.x, position.y, position.z, 1.0f);
    dir = transforms * dir;
    pos = transforms * pos;
    float xc = location.x;
    float yc = location.y;
    float zc = location.z;
    float x0 = position.x;
    float y0 = position.y;
    float z0 = position.z;
    float xd = direction.x;
    float yd = direction.y;
    float zd = direction.z;
    ret.viewVector.x = -direction.x;
    ret.viewVector.y = -direction.y;
    ret.viewVector.z = -direction.z;
    ret.hitMark.x = x0 + xd*t0;
    ret.hitMark.y = y0 + yd*t0;
    ret.hitMark.z = z0 + zd*t0;

    Vector objHit;
    objHit.x = pos[0] + dir[0]* t0;
    objHit.y = pos[1] + dir[1]* t0;
    objHit.z = pos[2] + dir[2]* t0;

    ret.normal.x = (objHit.x - xc)/radius;
    ret.normal.y = (objHit.y - yc)/radius;
    ret.normal.z = (objHit.z - zc)/radius;
    ret.normal = ret.normal.unit();
    glm::vec4 n = glm::vec4( ret.normal.x, ret.normal.y, ret.normal.z, 1 );
    n = transpose * n;
    ret.normal.x = n[0];
    ret.normal.y = n[1];
    ret.normal.z = n[2];
    ret.normal = ret.normal.unit();

    ret.camDistance = position.distance( ret.hitMark );
    ret.hit = true;
    ret.obj = this;
    return ret;
}
float Sphere::hitTest( Vector direction, Vector position )
{
    direction = direction.unit();
    glm::vec4 dir = glm::vec4(direction.x, direction.y, direction.z, 0.0f);
    glm::vec4 pos = glm::vec4(position.x, position.y, position.z, 1.0f);
    dir = transforms*dir;
    pos = transforms*pos;
    float xc = location.x;
    float yc = location.y;
    float zc = location.z;
    float x0 = pos[0];
    float y0 = pos[1];
    float z0 = pos[2];
    float xd = dir[0];
    float yd = dir[1];
    float zd = dir[2];

    float A = xd*xd + yd*yd + zd*zd;
    float B = 2*(xd *(x0-xc) + yd*(y0-yc) + zd*(z0-zc));
    float C = (x0-xc)*(x0-xc) + (y0-yc)*(y0-yc) + (z0-zc)*(z0-zc) - radius*radius;
    float disc = B*B -4*A*C;
    if(disc < .0001)
        return -1;

    float t0 = (-B - sqrt(disc))/2;
    if ( t0 < 0.001)
    {
        t0 = (-B + sqrt(disc))/2;
    }
    if( t0 <= 0.001 )// && t0 >= -.00001 )
        return -1;
    return t0;
}

cuda_sphere_t Sphere::getCudaSphere()
{
    cuda_sphere_t ret;
    ret.x = location.x;
    ret.y = location.y;
    ret.z = location.z;
    ret.r = radius;
    return ret;
}

int Sphere::parse( FILE *file )
{
    char cur = '\0';

    //location
    while(cur != '<')
    {
        if(fscanf(file, "%c", &cur) == EOF)
            return 1;
    }
    if( fscanf(file, " %f, %f, %f ", &(location.x), &(location.y), &(location.z) ) == EOF )
        return 1;

    //printf( "location: %f %f %f\n", location.x, location.y, location.z );
    cur = '\0';

    //radius
    //Read in everything until , so next item is radius
    while( cur != ',' )
    {
        if(fscanf( file, "%c", &cur) == EOF)
            return 1;
    }
    if( fscanf(file, "%f", &radius) == EOF )
        return 1;
    //printf( "radius: %f \n", radius );


    if(parsePigment( file ))
        return 1;
    if(parseFinish( file ))
        return 1;

    if ( parseTransforms( file ) )
        return 1;

    //Construct Bounding Box
    Vector min;
    min.x = location.x - radius;
    min.y = location.y - radius;
    min.z = location.z - radius;
    Vector max;
    max.x = location.x + radius;
    max.y = location.y + radius;
    max.z = location.z + radius;

    transformMinMax( &min, &max );
    
    boundingbox = new BoundingBox( min, max );

    return 0;
}

