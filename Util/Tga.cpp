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
    data = (Object::pixel **) malloc( sizeof(Object::pixel *) * height );
    for( int i = 0; i < height; i++ )
    {
        data[i] = (Object::pixel *) malloc( sizeof(Object::pixel) * width );
        for( int j = 0; j < width; j++ )
        {
            data[i][j].r = 0;
            data[i][j].g = 0;
            data[i][j].b = 0;
        }
    }
}
Tga::~Tga()
{
    for( int i = 0; i < height; i++ )
    {
        free( data[i] );
    }
    free( data );
    free( header );
}
void Tga::setPixel( int w, int h, Object::pixel p )
{
    data[h][w].r += p.r * .25;
    data[h][w].g += p.g * .25;
    data[h][w].b += p.b * .25;
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
            data[i][j].r = pow( data[i][j].r, .7 );
            data[i][j].b = pow( data[i][j].b, .7 );
            data[i][j].g = pow( data[i][j].g, .7 );
            if (data[i][j].r > 1.0)
                data[i][j].r = 1.0;
            if (data[i][j].g > 1.0)
                data[i][j].g = 1.0;
            if (data[i][j].b > 1.0)
                data[i][j].b = 1.0;

            unsigned int red = data[i][j].r * 255;
            unsigned int green = data[i][j].g * 255;
            unsigned int blue = data[i][j].b * 255;
            outfile.write( reinterpret_cast<char*>(&(blue)), sizeof(char) );
            outfile.write( reinterpret_cast<char*>(&(green)), sizeof(char) );
            outfile.write( reinterpret_cast<char*>(&(red)), sizeof(char) );
        }
    }
    outfile.close();
    return 0;
}
