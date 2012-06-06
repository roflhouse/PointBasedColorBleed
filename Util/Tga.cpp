/**
 *  CPE 2010
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */
#include "Tga.h"

Tga::Tga(short int w, short int h)
{
    width = w;
    height = h;
    header = new Header( width, height );
    data = (Color *) malloc( sizeof(Color) * height * width );
    if( data == NULL )
    {
       printf("Failed to malloc TGA\n");
       exit( 1);
    }
    for( int i = 0; i < height; i++ )
    {
        for( int j = 0; j < width; j++ ){
           data[i*width + j].r = 0;
           data[i*width + j].b = 0;
           data[i*width + j].g = 0;
        }
    }
}
Tga::~Tga()
{
    free( data );
    delete header ;
}
Color *Tga::getBuffer( )
{
   return data;
}
int Tga::getWidth( )
{
   return width;
}
int Tga::getHeight( )
{
   return height;
}
void Tga::setPixel( int w, int h, Color p )
{
    data[h * width + w] = p;
}
int Tga::writeTga( std::string filename )
{
    std::ofstream outfile(filename.c_str());
    header->writeHeader( &outfile );

    for( int i = 0; i < height; i++ )
    {
        for( int j = 0; j < width; j++ )
        {
            //Gamma Correction
            /*data[i][j].r = pow( data[i][j].r, .7 );
            data[i][j].b = pow( data[i][j].b, .7 );
            data[i][j].g = pow( data[i][j].g, .7 );
            if (data[i][j].r > 1.0)
                data[i][j].r = 1.0;
            if (data[i][j].g > 1.0)
                data[i][j].g = 1.0;
            if (data[i][j].b > 1.0)
                data[i][j].b = 1.0;
                */

            unsigned int red = data[i*width + j].r * 255;
            unsigned int green = data[i*width + j].g * 255;
            unsigned int blue = data[i*width + j].b * 255;
            outfile.write( reinterpret_cast<char*>(&(blue)), sizeof(char) );
            outfile.write( reinterpret_cast<char*>(&(green)), sizeof(char) );
            outfile.write( reinterpret_cast<char*>(&(red)), sizeof(char) );
        }
    }
    outfile.close();
    return 0;
}
