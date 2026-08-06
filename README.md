# BUCCS Notebooks

Machine-learning fundamentals on how to handle urban temperature measurement network data, from data processing to training models, developed for the Bochum Urban Climate Summer School (2026).


## Installation

conda activate <your-env>          
pip install -e .

<your-env> should contain the following basic packages: 
pandas
matplotlib
contextily
numpy
pyproj
cmcrameri
sklearn
rasterio
xgboost


## Data

Station observations from:
Dortmund: Hüser, C., Wolf, L., Gottschalk, N., Kittner, J., Kraas, B., Mittelstädt, C., Reinhart, V., Sismanidis, P., Wawrzyniak, N., & Bechtel, B. (2026). Data2Resilience - A Biometeorological Weather Station Network in Dortmund (1.0.0). Zenodo. https://doi.org/10.5281/zenodo.18221203.

Bern, Freiburg, Ghent: Amini, S., Huerta, A., Franke, J., Brugnara, Y., Caluwaerts, S., Anet, J., Savi´c, S., Gubler, M., Steeneveld, G.-J., Chapman, L., Meier, F., Dubreuil, V., Christen, A., Zeeman, M., Lali´c, B., Schl¨ogl, S., K¨ayhk¨o, J., Azadfar, A., & Br¨onnimann, S. (2026). Comprehensive compilation and quality assessment of street-level urban air temperature measurements across European networks. Scientific Data 2026.
https://doi.org/10.1038/s41597-026-06804-4sources


“Generated using European Union's Copernicus Land Monitoring Service information; 
TCD: https://doi.org/10.2909/e677441e-fb94-431c-b4f9-304f10e4dfd8
Imperviousness density: https://doi.org/10.2909/34ef6334-d432-4041-a3da-67e156d6501d 
Building Block Heights: https://doi.org/10.2909/42690e05-edf4-43fc-8020-33e130f62023 ”

Local Climate Zones from: Demuzere, M., Bechtel, B., Middel, A., & Mills, G. (2019). Mapping Europe into local climate zones. PLOS ONE, 14(4), e0214474. https://doi.org/10.1371/journal.pone.0214474

Digital Terrain Model from: Hengl, Tomislav, Leal Parente, Leandro, Krizan, Josip, and Bonannella, Carmelo. 2020. "Continental Europe Digital Terrain Model at 30 M Resolution Based on GEDI, Icesat-2, AW3D, GLO-30, EUDEM, MERIT DEM and Background Layers." Zenodo. https://doi.org/10.5281/zenodo.4724549.


 
Met data from:
Copernicus Climate Change Service, Climate Data Store, (2023): ERA5 hourly data on single levels from 1940 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). DOI: 10.24381/cds.adbb2d47 (Accessed on 06-Aug-2026)
Copernicus Climate Change Service (C3S)(2019): ERA5-Land hourly data from 1950 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). DOI: 10.24381/cds.e2161bac (Accessed on 06-Aug-2026)




## Citation

If you use this code, please cite it via the "Cite this repository" button
(GitHub sidebar) or the `CITATION.cff` file.

## License

Released under the MIT License — see [LICENSE](LICENSE).

## Authors

Charles Pierce, University of Bern 
Sara Top, University of Ghent / KMI
...