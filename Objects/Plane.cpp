/**
 *  CPE 2010
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */
#include "Plane.h"

Plane::Plane()
{
}
Object::rayInfo Plane::rayIntersect( Vector direction, Vector position, float t )
{
    Object::rayInfo ret;
    ret.obj = this;
    ret.hit = true;
    ret.viewVector.x = -direction.x;
    ret.viewVector.y = -direction.y;
    ret.viewVector.z = -direction.z;

    ret.hitMark.x = position.x + direction.x*t;
    ret.hitMark.y = position.y + direction.y*t;
    ret.hitMark.z = position.z + direction.z*t;
    ret.camDistance = position.distance( ret.hitMark );
    ret.normal = normal;
    return ret;
}
float Plane::hitTest( Vector direction, Vector position )
{
    direction = direction.unit();
    glm::vec4 dir = glm::vec4(direction.x, direction.y, direction.z, 0.0f);
    glm::vec4 pos = glm::vec4(position.x, position.y, position.z, 1.0f);
    dir = transform*dir;
    pos = transform*pos;
    direction.x = dir[0];
    direction.y = dir[1];
    direction.z = dir[2];
    position.x = pos[0];
    position.y = pos[1];
    position.z = pos[2];


    float vd = normal.dot(direction);
    if((distance < 0 && vd > -0.0001) || (distance > 0 && vd < 0.0001))
        return -1;
    float v0 = (point.newDirection(position).dot(normal));
    float t = v0/vd;
    //make sure its pointing right directions
    if( t < 0 )
        t = -t;
    if( t < 0.001)
        return -1;
    return t;
}
int Plane::parse( FILE *file )
{
    char cur = '\0';
    //location
    while(cur != '<')
    {
        if(fscanf(file, "%c", &cur) == EOF)
            return 1;
    }
    if( fscanf(file, " %f , %f , %f ", &(normal.x), &(normal.y), &(normal.z) ) == EOF )
        return 1;
    //printf( "normal: %f %f %f\n", normal.x, normal.y, normal.z );

    cur = '\0';
    //distance
    //Read in everything until , so next item is distance
    while( cur != ',' )
    {
        if(fscanf( file, "%c", &cur) == EOF)
            return 1;
    }
    if( fscanf(file, "%f", &distance) == EOF )
        return 1;
    //printf( "distance: %f \n", distance );
    distance = distance / normal.mag();
    normal = normal.unit();
    //A != 0
    if( normal.x > 0.0001 || normal.x < -0.0001 )
    {
        point.x = distance / normal.x;
        point.y = 0;
        point.z = 0;
    }
    //B != 0
    else if( normal.y > 0.0001 || normal.y < -0.0001 )
    {
        point.x = 0;
        point.y = distance / normal.y;
        point.z = 0;
    }
    //C != 0
    else if( normal.z > 0.0001 || normal.z < -0.0001 )
    {
        point.x = 0;
        point.y = 0;
        point.z = distance / normal.z;
    }
    else
    {
        printf("Plane normal not valid\n");
        exit(1);
    }
    if(parsePigment( file ))
        return 1;
    if(parseFinish( file ))
        return 1;
    if(parseTransforms( file ))
        return 1;

    glm::vec4 n = glm::vec4( normal.x, normal.y, normal.z, 1 );

    n = transpose * n ;
    normal.x = n[0];
    normal.y = n[1];
    normal.z = n[2];
    normal = normal.unit();

    //Parsing transforms uses up the ending bracket so no need to read to it
    return 0;
}


