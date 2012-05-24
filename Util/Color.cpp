#include "Color.h"
#include <stdio.h>

Color limitColor( const Color &in )
{
   Color ret;
   if( in.r > 1.0 )
      ret.r = 1.0;
   else if( in.r < 0.0 )
      ret.r = 0;
   else
      ret.r = in.r;

   if( in.g > 1.0 )
      ret.g = 1.0;
   else if( in.g < 0.0 )
      ret.g = 0;
   else 
      ret.g = in.g;

   if( in.b > 1.0 )
      ret.b = 1.0;
   else if( in.b < 0.0 )
      ret.b = 0;
   else
      ret.b = in.b;

   return ret;
}
Color plus( const Color &first, const Color &other )
{
   Color ret;
   ret.r = first.r + other.r;
   ret.g = first.g + other.g;
   ret.b = first.b + other.b;
   return limitColor( ret );
}
