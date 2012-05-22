/**
 *  CPE 2011
 *  -------------------
 *  Program 
 *
 *  Last Modified: 
 *  @author Nick Feeney
 */
#ifndef BOUNDINGBOX_H
#define BOUNDINGBOX_H
#include <sys/types.h>
#include <unistd.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include "Vector.h"

class BoundingBox
{
    public:
        BoundingBox( Vector minIn, Vector maxIn );
        BoundingBox( BoundingBox one, BoundingBox two );

        bool testForHit( Vector dir, Vector pos );

        Vector min;
        Vector max;
};
#endif
