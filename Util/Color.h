/**
 *  CPE 2012
 *  -------------------
 *  Program 
 *
 *  Last Modified: 
 *  @author Nick Feeney
 */

#ifndef COLOR_H
#define COLOR_H
typedef struct Color {
   float r;
   float g;
   float b;
} Color;

Color limitColor( const Color &in );
Color plus( const Color &first, const Color &other );
#endif
