/**
 *  CPE 2010
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */

#include "LightSource.h"

LightSource::LightSource()
{
}
int LightSource::parse( FILE *file )
{
    char cur = '\0';
    //location
    while(cur != '<')
    {
        //read in until data
        if( fscanf(file, "%c", &cur) == EOF)
            return 1;
    }
    if( fscanf(file, " %f , %f , %f ", &(location.x), &(location.y), &(location.z) ) == EOF )
        return 1;

    //printf( " location: %f, %f, %f\n", location.x, location.y, location.z );
    cur = '\0';

    while(cur != '<')
    {
        //read in until rgb data
        if( fscanf(file, "%c", &cur) == EOF)
            return 1;
    }

    if( fscanf(file, " %f , %f , %f ", &(red), &(green), &(blue) ) == EOF )
        return 1;

    //printf( " color: %f, %f, %f\n", red, green, blue );
    while(cur != '}')
    {
        if( fscanf(file, "%c", &cur) == EOF)
            return 1;
    }
    return 0;
}
