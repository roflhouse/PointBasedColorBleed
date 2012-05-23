/**
 *  CPE 2010
 *  -------------------
 *  Program 
 *
 *  Last Modified: 
 *  @author Nick Feeney
 */
#ifndef TGA_H
#define TGA_H
#include <stdlib.h>
#include <string>
#include <stdint.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <iostream>
#include <fstream>
#include <math.h>

#include "Header.h"
#include "Color.h"
class Tga
{
    public:
        Tga( short int w, short int h );
        ~Tga();
        int writeTga(std::string filename);
        void setPixel(int width, int height, Color p);
    private:
        Color **data;
        Header *header;
        short int width;
        short int height;
};
#endif
