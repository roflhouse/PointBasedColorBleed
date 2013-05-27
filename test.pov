// cs174, assignment 1 sample file (RIGHT HANDED)

camera {
  location  <0, 0, 17>
  up        <0,  1,  0>
  right     <1, 0,  0>
  look_at   <0, 0, 0>
}


light_source {<-7, 3, 0> color rgb <7, 7, 7>}

sphere { <2, 0, -2>, 2
  pigment { color rgb <1.0, 0.0, 0.0>}
  finish {ambient 0.0 diffuse 0.8 specular 0.0 }
}

plane { <0,1,0>, -2
   pigment { color rgb <0, 1.0, 0> }
   finish {ambient 0.0 diffuse 0.8 }
}
