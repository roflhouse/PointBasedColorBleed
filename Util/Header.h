/**
 *  CPE 2010
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */
#ifndef HEADER_H
#define HEADER_H

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdint.h>
#include <string>
#include <iostream>
#include <fstream>

class Header
{
    public:
        typedef struct {
            unsigned char  idlength;
            unsigned char  colourmaptype;
            unsigned char  datatypecode;
            unsigned char  colourmaporigin1;
            unsigned char  colourmaporigin2;
            unsigned char  colourmaplength1;
            unsigned char  colourmaplength2;
            unsigned char  colourmapdepth;
            unsigned char x_origin1;
            unsigned char x_origin2;
            unsigned char y_origin1;
            unsigned char y_origin2;
            unsigned char width1;
            unsigned char width2;
            unsigned char height1;
            unsigned char height2;
            unsigned char  bitsperpixel;
            unsigned char  imagedescriptor;
        } Header_Struct;

        Header( short int width, short int height );
        int writeHeader( std::ofstream *out );
    private:
        Header_Struct header;
};
#endif
