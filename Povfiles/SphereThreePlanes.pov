// cs174, assignment 1 sample file (RIGHT HANDED)

camera {
  location  <0, 0, 14>
  up        <0,  1,  0>
  right     <1.33333, 0,  0>
  look_at   <0, 0, 0>
}


light_source {<100, 100, 100> color rgb <1.5, 1.5, 1.5>}

sphere { <0, 0, 0>, 3
  pigment { color rgb <0.5, 0.5, 0.5>}
  finish {ambient 0.2 diffuse 0.4}
  translate< -4 , 0 , 0>
}

plane {<0, 1, 0>, -4
      pigment {color rgb <0, 0, 1.0>}
      finish {ambient 0.4 diffuse 0.8}
}

plane {<1, 0, 0>, -7
      pigment {color rgb <1, 0, 0>}
      finish {ambient 0.4 diffuse 0.8}
      rotate< 0, 5, 0 >
}

plane {<0, 0, 1>, -7
      pigment {color rgb <0, 1, >}
      finish {ambient 0.4 diffuse 0.8}
}
