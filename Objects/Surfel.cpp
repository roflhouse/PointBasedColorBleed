/**
 *  CPE 2010
 *  -------------------
 *  Program 
 *
 *  Last Modified: 
 *  @author Nick Feeney
 */

#include "Surfel.h"


SurfelArray createSurfelArray()
{
   SurfelArray IA;
   IA.array = (Surfel *) malloc( sizeof(Surfel) * 1000 );
   IA.num = 0;
   IA.max = 1000;
   return IA;
}
void growSA( SurfelArray &in )
{
   in.max *= 5;
   in.array = realloc( in.array, sizeof(Surfel) * in.max );
   if( in.array = NULL )
   {
      printf("You have run out of memory\n");
      exit(1);
   }
}
void shrinkSA( SurfelArray &in )
{
   in.max = in.num;
   in.array = realloc( in.array, sizeof(Surfel) * in.max );
   if( in.array = NULL )
   {
      printf("You have run out of memory\n");
      exit(1);
   }
}
void addToSA( SurfelArray &in, const Surfel &surfel )
{
   if( in.num +1 >=in.max )
   {
      growIA( in );
   }
   in.array[in.num] = surfel;
   in.num++;
}
void freeSurfelArray( SurfelArray &array )
{
   free( array.array );
}
