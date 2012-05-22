/**
 *  CPE 2010
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */
#include "Header.h"
Header::Header( short int w, short int h )
{
    header.idlength = 0;
    header.colourmaptype = 0;
    header.datatypecode = 2;
    header.colourmaptype = 0;
    header.colourmaporigin1 = 0;
    header.colourmaporigin2 = 0;
    header.colourmaplength1 = 0;
    header.colourmaplength2 = 0;
    header.colourmapdepth = 0;
    header.x_origin1 = 0;
    header.x_origin2 = 0;
    header.y_origin1 = 0;
    header.y_origin2 = 0;
    header.width1 = (w & 0x00FF);
    header.width2 = (w & 0xFF00)/256;
    header.height1 = (h & 0x00FF);
    header.height2 = (h & 0xFF00)/256;
    header.bitsperpixel = 24;
    header.imagedescriptor = 0;
}
int Header::writeHeader( std::ofstream *out )
{
    out->write( reinterpret_cast<char*>(&header), sizeof(header) );

    /*
    int temp = 0;

    //idlength
    out->write( reinterpret_cast<char*>(&temp), sizeof(char) );
    //colourmaptype
    out->write( reinterpret_cast<char*>(&temp), sizeof(char) );
    //datatypecode
    temp = 2;
    out->write( reinterpret_cast<char*>(&temp), sizeof(char) );
    //colourmaptype
    temp = 0;
    out->write( reinterpret_cast<char*>(&temp), sizeof(char) );
    out->write( reinterpret_cast<char*>(&temp), sizeof(char) );
    //colourmaplength
    out->write( reinterpret_cast<char*>(&temp), sizeof(char) );
    out->write( reinterpret_cast<char*>(&temp), sizeof(char) );
    //colourmapdepth
    out->write( reinterpret_cast<char*>(&temp), sizeof(char) );
    //x_origin
    out->write( reinterpret_cast<char*>(&temp), sizeof(char) );
    out->write( reinterpret_cast<char*>(&temp), sizeof(char) );
    //y_origin
    out->write( reinterpret_cast<char*>(&temp), sizeof(char) );
    out->write( reinterpret_cast<char*>(&temp), sizeof(char) );
    //width
    temp = header.width & 0x00FF;
    out->write( reinterpret_cast<char*>(&temp), sizeof(char) );
    temp = (header.width & 0xFF00)/256;
    out->write( reinterpret_cast<char*>(&temp), sizeof(char) );
    //height
    temp = header.height & 0x00FF;
    out->write( reinterpret_cast<char*>(&temp), sizeof(char) );
    temp = (header.height & 0xFF00)/256;
    out->write( reinterpret_cast<char*>(&temp), sizeof(char) );
    //bitsperpixel
    temp = 24;
    out->write( reinterpret_cast<char*>(&temp), sizeof(char) );
    //imagedescriptor
    temp = 0;
    out->write( reinterpret_cast<char*>(&temp), sizeof(char) );
*/
    return sizeof(header);
}
