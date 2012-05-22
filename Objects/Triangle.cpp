/**
 *  CPE 2010
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */
#include "Triangle.h"

Triangle::Triangle()
{
}
Object::rayInfo Triangle::rayIntersect( Vector direction, Vector position, float t )
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
float Triangle::hitTest( Vector direction, Vector position )
{
    direction = direction.unit();
    glm::vec4 dir = glm::vec4(direction.x, direction.y, direction.z, 0.0f);
    glm::vec4 pos = glm::vec4(position.x, position.y, position.z, 1.0f);
    dir = transforms*dir;
    pos = transforms*pos;
    direction.x = dir[0];
    direction.y = dir[1];
    direction.z = dir[2];
    position.x = pos[0];
    position.y = pos[1];
    position.z = pos[2];

    float aa = a.x - b.x;
    float bb = a.y - b.y;
    float cc = a.z - b.z;
    float d = a.x - c.x;
    float e = a.y - c.y;
    float f = a.z - c.z;
    float g = direction.x;
    float h = direction.y;
    float i = direction.z;
    float j = a.x - position.x;
    float k = a.y - position.y;
    float l = a.z - position.z;

    float t = -1;
    float beta = 0;
    float gamma = 0;

    float ei_m_hf = e*i - h*f;
    float gf_m_di = g*f - d*i;
    float dh_m_eg = d*h - e*g;
    float ak_m_jb = aa*k - j*bb;
    float jc_m_al = j*cc - aa*l;
    float bl_m_kc = bb*l - k*cc;
    float M = aa*ei_m_hf + bb*gf_m_di + cc*dh_m_eg;
    if( M  < 0.0001 && M > -0.0001 )
        return -1;
    t = -(f*ak_m_jb + e*jc_m_al + d* bl_m_kc)/M;

    if(t < 0.001)
        return -1;

    gamma = (i*ak_m_jb + h*jc_m_al + g* bl_m_kc)/M;
    if(gamma < 0 || gamma > 1)
        return -1;

    beta = (j*ei_m_hf + k*gf_m_di + l*dh_m_eg)/M;
    if(beta < 0 || beta > (1 - gamma))
        return -1;

    return t;
}

cuda_triangle_t Triangle::getCudaTriangle()
{

    cuda_triangle_t ret;
    ret.ax = a.x;
    ret.ay = a.y;
    ret.az = a.z;
    ret.bx = b.x;
    ret.by = b.y;
    ret.bz = b.z;
    ret.cx = c.x;
    ret.cy = c.y;
    ret.cz = c.z;

    return ret;
}

int Triangle::parse( FILE *file )
{
    char cur = '\0';
    //Point a
    while(cur != '<')
    {
        if(fscanf(file, "%c", &cur) == EOF)
            return 1;
    }
    if( fscanf(file, " %f , %f , %f ", &(a.x), &(a.y), &(a.z) ) == EOF )
        return 1;
    //printf( "Triangle a: %f %f %f\n", a.x, a.y, a.z );

    cur = '\0';

    //Point b
    while(cur != '<')
    {
        if(fscanf(file, "%c", &cur) == EOF)
            return 1;
    }
    if( fscanf(file, " %f , %f , %f ", &(b.x), &(b.y), &(b.z) ) == EOF )
        return 1;
    //printf( "Triangle b: %f %f %f\n", b.x, b.y, b.z );

    cur = '\0';

    //Point c
    while(cur != '<')
    {
        if(fscanf(file, "%c", &cur) == EOF)
            return 1;
    }
    if( fscanf(file, " %f , %f , %f ", &(c.x), &(c.y), &(c.z) ) == EOF )
        return 1;
    //printf( "Triangle c: %f %f %f\n", c.x, c.y, c.z );

    Vector atob = b.newDirection( a );
    Vector atoc = c.newDirection( a );
    normal = atob.cross( atoc ).unit();

    if(parsePigment( file ))
        return 1;
    if(parseFinish( file ))
        return 1;
    if(parseTransforms( file ))
        return 1;
    normal = normal.unit();
    glm::vec4 n = glm::vec4( normal.x, normal.y, normal.z, 1 );

    n = transpose * n ;
    normal.x = n[0];
    normal.y = n[1];
    normal.z = n[2];
    normal = normal.unit();

    //Construct Bouning Box
    Vector min;
    Vector max;
    min = a;
    max = a;

    if( min.x > b.x )
        min.x = b.x;
    if( min.y > b.y )
        min.y = b.y;
    if( min.z > b.z )
        min.z = b.z;
    if( min.x > c.x )
        min.x = c.x;
    if( min.y > c.y )
        min.y = c.y;
    if( min.z > c.z )
        min.z = c.z;

    if( max.x < b.x )
        max.x = b.x;
    if( max.y < b.y )
        max.y = b.y;
    if( max.z < b.z )
        max.z = b.z;
    if( max.x < c.x )
        max.x = c.x;
    if( max.y < c.y )
        max.y = c.y;
    if( max.z < c.z )
        max.z = c.z;

    transformMinMax( &min, &max );

    boundingbox = new BoundingBox( min, max );

    //Parsing transforms uses up the ending bracket so no need to read to it
    return 0;
}
