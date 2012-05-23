/**
 *  CPE 2010
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */

#ifndef PARSER_H
#define PARSER_H
#include <vector>
#include <stdio.h>
#include <string>

#include "../Objects/Objects.h"
#include "vec3.h"
#include "Scene.h"

Scene parseFile( std::string file );
#endif
