camera {
   location  <0, 5, 15>
      up        <0,  1,  0>
      right     <1, 0,  0>
      look_at   <0, 4.9, 0>
}

light_source {<-3, 10, -3> color rgb <2, 2, 2>}

light_source {<3, 10, -3> color rgb <2, 2, 2>}

light_source {<0, 10, 3> color rgb <2, 2, 2>}


//left
triangle { <-5, 0, 5>, <-5, 10, -5>, <-5, 10, 5>,
   pigment { color rgb< .9, .00, .10>}
   finish{ ambient 0.0 diffuse 1.0 }
}
triangle { <-5, 0, 5.01>, <-5, 0, -5>, <-5, 10, -5>,
   pigment { color rgb< .9, .00, .10>}
   finish{ ambient 0.0 diffuse 1.0 }
}
//back
triangle { <-5, 10, -5>, <-5, 0, -5>, <5,10,-5>,
   pigment { color rgb< .7, .6, .5>}
   finish{ ambient 0.0 diffuse 1.0 }
}
triangle { <-5, 0, -5>, <5, 0, -5>, <5, 10, -5>,
   pigment { color rgb< .7, .6, .5>}
   finish{ ambient 0.0 diffuse 1.0 }
}
//right
triangle { <5, 0, -5>, <5, 10, 5>, <5, 10, -5>,
   pigment { color rgb< .00, .9, .10>}
   finish{ ambient 0.0 diffuse 1.0 }
}
triangle { <5, 0, -5.00>, <5, 0,5>, <5, 10, 5>,
   pigment { color rgb< .0, .9, .1>}
   finish{ ambient 0.0 diffuse 1.0 }
}
//bottom
triangle { <-5, 0, 5>, <5, 0, -5>, <-5, 0, -5>,
   pigment { color rgb< .7, .6, .5>}
   finish{ ambient 0.0 diffuse 1.0 }
}
triangle { <-5, 0,5>, <5, 0, 5>, <5, 0, -5.01>,
   pigment { color rgb< .7, .6, .5>}
   finish{ ambient 0.0 diffuse 1.0 }
}

sphere { <-2.5, 2, -2>, 2
   pigment { color rgb <0.6, 0.6, 0.6>}
   finish {ambient 0.0 diffuse 0.8 specular 0.0 }
}
sphere { <2.5, 2, 0>, 2
   pigment { color rgb <0.6, 0.6, 0.6>}
   finish {ambient 0.0 diffuse 0.8 specular 0.0 }
}
