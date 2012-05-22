/**
 *  CPE 2010
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */

#include "Parser.h"
#include <stdlib.h>

Parser::Parser( std::string filename )
{
    //Open file for writing
    FILE *file = fopen(filename.c_str(), "r");
    if(file == NULL)
    {
        printf("Error Occured opening file\n");
        exit(EXIT_FAILURE);
    }

    while(1)
    {
        //starting off eating all whitespace
        char cur = ' ';
        while( isspace(cur) )
        {
            if( fscanf( file, "%c", &cur ) == EOF )
            {
                printf("End of File reached\n");
                //Construct bvh Now that the input is finished
                bvh = new BVH();
                printf("BVH Created\n");
                return;
            }
        }

        //check for comment
        if(cur == '/')
        {
            while(cur != '\n')
            {
                if( fscanf( file, "%c", &cur ) == EOF )
                {
                    printf("Error Occured reading file\n");
                    exit(EXIT_FAILURE);
                }
            }
        }
        else if( cur == 'c' || cur == 'C' )
        {
            camera = new Camera;
            if ( camera->parse( file ) )
            {
                printf("Error Parsing camera\n");
                exit(EXIT_FAILURE);
            }
        }
        else if( cur == 'l' || cur == 'L' )
        {
            LightSource *light = new LightSource;
            if( light->parse( file ) )
            {
                printf("Error parsing lightsource\n");
                exit(EXIT_FAILURE);
            }
            if( numLights + 1 >= maxLights )
            {
                maxLights = maxLights*1000;
                lights = (LightSource **)realloc( lights, sizeof(LightSource *)*maxLights );
            }
            lights[numLights] = light;
            numLights++;
        }
        else if( cur == 's' || cur == 'S' )
        {
            Sphere *sph = new Sphere;
            if( sph->parse( file ) )
            {
                printf("Error parsing Sphere\n");
                exit(EXIT_FAILURE);
            }
            if( numSpheres+1 >= maxSpheres )
            {
                maxSpheres = maxSpheres*1000;
                spheres = (Object **) realloc( spheres, sizeof(Object *) * maxSpheres );
            }
            spheres[numSpheres] = sph;
            numSpheres++;
        }
        else if( cur == 'p' || cur == 'P' )
        {
            Plane *plane = new Plane;
            if( plane->parse( file ))
            {
                printf( "Error parsing Plane\n");
                exit(EXIT_FAILURE);
            }
            if( numPlanes+1 >= maxPlanes )
            {
                maxPlanes = maxPlanes*1000;
                planes = (Object **) realloc( planes, maxPlanes * sizeof(Object *) );
            }
            planes[numPlanes] = plane;
            numPlanes++;
        }
        else if( cur == 't' || cur == 'T' )
        {
            Triangle *tri = new Triangle;
            if( tri->parse( file ))
            {
                printf( "Error parsing Plane\n");
                exit(EXIT_FAILURE);
            }
            if( numTriangles+1 >= maxTriangles )
            {
                maxTriangles = maxTriangles*1000;
                triangles = (Object **) realloc( triangles, maxTriangles * sizeof(Object *) );
            }
            triangles[numTriangles] = tri;
            numTriangles++;
        }
        else
        {
            printf("Unknown Keyword Failure char was |%c|\n", cur);
            exit(EXIT_FAILURE);
        }
    }
}
