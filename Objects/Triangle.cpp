/**
 *  CPE 2010
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */
#include "Triangle.h"

Triangle parseTriangle( FILE *file )
{
   Triangle tri;
    char cur = '\0';
    //Point a
    while(cur != '<')
    {
        if(fscanf(file, "%c", &cur) == EOF)
        {
           printf("Error Parsing Triangle\n");
           exit(1);
        }
    }
    if( fscanf(file, " %f , %f , %f ", &(tri.a.x), &(tri.a.y), &(tri.a.z) ) == EOF )
        {
           printf("Error Parsing Triangle\n");
           exit(1);
        }
    printf( "Triangle a: %f %f %f\n", tri.a.x, tri.a.y, tri.a.z );

    cur = '\0';

    //Point b
    while(cur != '<')
    {
        if(fscanf(file, "%c", &cur) == EOF)
        {
           printf("Error Parsing Triangle\n");
           exit(1);
        }
    }
    if( fscanf(file, " %f , %f , %f ", &(tri.b.x), &(tri.b.y), &(tri.b.z) ) == EOF )
        {
           printf("Error Parsing Triangle\n");
           exit(1);
        }
    printf( "Triangle b: %f %f %f\n", tri.b.x, tri.b.y, tri.b.z );

    cur = '\0';

    //Point c
    while(cur != '<')
    {
        if(fscanf(file, "%c", &cur) == EOF)
        {
           printf("Error Parsing Triangle\n");
           exit(1);
        }
    }
    if( fscanf(file, " %f , %f , %f ", &(tri.c.x), &(tri.c.y), &(tri.c.z) ) == EOF )
        {
           printf("Error Parsing Triangle\n");
           exit(1);
        }
    printf( "Triangle c: %f %f %f\n", tri.c.x, tri.c.y, tri.c.z );

    vec3 atob = newDirection(tri.b, tri.a);
    vec3 atoc = newDirection( tri.c, tri.a );
    normal = unit(cross( atob, atoc ));

    parsePigment( file );
    parseFinish( file );
    parseTransforms( file );
    normal = normal.unit();
    glm::vec4 n = glm::vec4( normal.x, normal.y, normal.z, 1 );

    n = transpose * n ;
    normal.x = n[0];
    normal.y = n[1];
    normal.z = n[2];
    normal = normal.unit();

    //Parsing transforms uses up the ending bracket so no need to read to it
    return tri;
}
