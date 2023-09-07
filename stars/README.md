# Prediction of star class with machine learning

This data was sampled from a catalog of stars observed with LAMOST as part of the LEAGUE sky survey. 

Each row corresponds to a single star, and the following attributes are given for each:

| Variable Name    | Explanation                                                                              |
|----------|--------------------------------------------------------------------------------------------------|
| obsid    | Unique identifier number for the star                                                            |
| obsdate  | Observation date       (dd/mm/yyyy)                                                                          |
| ra       | Right ascension, astronomical coordinate analogous to the x direction (Degree)                            |
| dec      | Declination, astronomical coordinate analogous to the y direction (Degree)                                |
| subclass | Luminosity* (first lowercase letters if exists) and Spectral** (capital letter and digit) classes           |
| mag5     | Measured brightness of the star in the infrared (Magnitude)                                                 |
| z        | Estimated redshift amount (Fraction change)                                                                                |
| z_err    | Window of uncertainty for redshift                                                                |
| rv       | Radial velocity (along line of sight) of the star estimated by the doppler effect (Km/s)                                   |
| rv_err   | Window of uncertainty for relative velocity                                                      |
| logg     | Estimated surface gravity in logarithmic scale (Dex, $10^x$, compared to the Sun)                                                   |
| logg_err | Window of uncertainty for surface gravity                                                        |
| teff     | Estimated effective temperature of the surface (Kelvin)                                                  |
| teff_err | Window of uncertainty for surface temperature                                                    |
| feh      | Estimated element composition for elements heavier than hydrogen (Dex, $10^x$, compared to the Sun)                                |
| feh_err  | Window of uncertainty for heavier element composition                                            |

(*) Luminosity classes are indicated by designations g for giant, d for dwarf (like our Sun) and sd for subdwarf. <br>
(**) Spectral classes are made up of the main class (the letter) and the digit portion allows for smooth transition. (A star right in-between A and F is labeled A5.)
