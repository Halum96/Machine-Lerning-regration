
import datetime
from netCDF4 import num2date
import numpy as np
import pandas as pd
from requests.exceptions import HTTPError
from xml.etree.ElementTree import ParseError

from pvlib.location import Location
from pvlib.irradiance import liujordan, extraradiation, disc
from siphon.catalog import TDSCatalog
from siphon.ncss import NCSS

import warnings

warnings.warn(
    'The forecast module algorithms and features are highly experimental. ' +
    'The API may change, the functionality may be consolidated into an io ' +
    'module, or the module may be separated into its own package.')


from functools import partial
import warnings
import pandas as pd

from pvlib import (solarposition, pvsystem, clearsky, atmosphere, tools)
from pvlib.tracking import SingleAxisTracker
import pvlib.irradiance  # avoid name conflict with full import


def basic_chain(times, latitude, longitude,
                module_parameters, inverter_parameters,
                irradiance=None, weather=None,
                surface_tilt=None, surface_azimuth=None,
                orientation_strategy=None,
                transposition_model='haydavies',
                solar_position_method='nrel_numpy',
                airmass_model='kastenyoung1989',
                altitude=None, pressure=None,
                **kwargs):


    # use surface_tilt and surface_azimuth if provided,
    # otherwise set them using the orientation_strategy
    if surface_tilt is not None and surface_azimuth is not None:
        pass
    elif orientation_strategy is not None:
        surface_tilt, surface_azimuth = \
            get_orientation(orientation_strategy, latitude=latitude)
    else:
        raise ValueError('orientation_strategy or surface_tilt and '
                         'surface_azimuth must be provided')

    times = times

    if altitude is None and pressure is None:
        altitude = 0.
        pressure = 101325.
    elif altitude is None:
        altitude = atmosphere.pres2alt(pressure)
    elif pressure is None:
        pressure = atmosphere.alt2pres(altitude)

    solar_position = solarposition.get_solarposition(times, latitude,
                                                     longitude,
                                                     altitude=altitude,
                                                     pressure=pressure,
                                                     method=solar_position_method,
                                                     **kwargs)

    # possible error with using apparent zenith with some models
    airmass = atmosphere.relativeairmass(solar_position['apparent_zenith'],
                                         model=airmass_model)
    airmass = atmosphere.absoluteairmass(airmass, pressure)
    dni_extra = pvlib.irradiance.extraradiation(solar_position.index)
    dni_extra = pd.Series(dni_extra, index=solar_position.index)

    aoi = pvlib.irradiance.aoi(surface_tilt, surface_azimuth,
                               solar_position['apparent_zenith'],
                               solar_position['azimuth'])

    if irradiance is None:
        linke_turbidity = clearsky.lookup_linke_turbidity(
            solar_position.index, latitude, longitude)
        irradiance = clearsky.ineichen(
            solar_position['apparent_zenith'],
            airmass,
            linke_turbidity,
            altitude=altitude,
            dni_extra=dni_extra
            )

    total_irrad = pvlib.irradiance.total_irrad(
        surface_tilt,
        surface_azimuth,
        solar_position['apparent_zenith'],
        solar_position['azimuth'],
        irradiance['dni'],
        irradiance['ghi'],
        irradiance['dhi'],
        model=transposition_model,
        dni_extra=dni_extra)

    if weather is None:
        weather = {'wind_speed': 0, 'temp_air': 20}

    temps = pvsystem.sapm_celltemp(total_irrad['poa_global'],
                                   weather['wind_speed'],
                                   weather['temp_air'])

    effective_irradiance = pvsystem.sapm_effective_irradiance(
        total_irrad['poa_direct'], total_irrad['poa_diffuse'], airmass, aoi,
        module_parameters)

    dc = pvsystem.sapm(effective_irradiance, temps['temp_cell'],
                       module_parameters)

    ac = pvsystem.snlinverter(dc['v_mp'], dc['p_mp'], inverter_parameters)

    return dc, ac


def get_orientation(strategy, **kwargs):
    if strategy == 'south_at_latitude_tilt':
        surface_azimuth = 180
        surface_tilt = kwargs['latitude']
    elif strategy == 'flat':
        surface_azimuth = 180
        surface_tilt = 0
    else:
        raise ValueError('invalid orientation strategy. strategy must '
                         'be one of south_at_latitude, flat,')

    return surface_tilt, surface_azimuth


class ModelChain(object):
 

    def __init__(self, system, location,
                 orientation_strategy=None,
                 clearsky_model='ineichen',
                 transposition_model='haydavies',
                 solar_position_method='nrel_numpy',
                 airmass_model='kastenyoung1989',
                 dc_model=None, ac_model=None, aoi_model=None,
                 spectral_model=None, temp_model='sapm',
                 losses_model='no_loss', name=None, **kwargs):

        self.name = name
        self.system = system
        self.location = location
        self.clearsky_model = clearsky_model
        self.transposition_model = transposition_model
        self.solar_position_method = solar_position_method
        self.airmass_model = airmass_model

        # calls setters
        self.dc_model = dc_model
        self.ac_model = ac_model
        self.aoi_model = aoi_model
        self.spectral_model = spectral_model
        self.temp_model = temp_model
        self.losses_model = losses_model
        self.orientation_strategy = orientation_strategy

        self.weather = None
        self.times = None
        self.solar_position = None

    def __repr__(self):
        attrs = [
            'name', 'orientation_strategy', 'clearsky_model',
            'transposition_model', 'solar_position_method',
            'airmass_model', 'dc_model', 'ac_model', 'aoi_model',
            'spectral_model', 'temp_model', 'losses_model'
            ]

        def getmcattr(self, attr):
            """needed to avoid recursion in property lookups"""
            out = getattr(self, attr)
            try:
                out = out.__name__
            except AttributeError:
                pass
            return out

        return ('ModelChain: \n  ' + '\n  '.join(
            ('{}: {}'.format(attr, getmcattr(self, attr)) for attr in attrs)))

    @property
    def orientation_strategy(self):
        return self._orientation_strategy

    @orientation_strategy.setter
    def orientation_strategy(self, strategy):
        if strategy == 'None':
            strategy = None

        if strategy is not None:
            self.system.surface_tilt, self.system.surface_azimuth = \
                get_orientation(strategy, latitude=self.location.latitude)

        self._orientation_strategy = strategy

    @property
    def dc_model(self):
        return self._dc_model

    @dc_model.setter
    def dc_model(self, model):
        if model is None:
            self._dc_model = self.infer_dc_model()
        elif isinstance(model, str):
            model = model.lower()
            if model == 'sapm':
                self._dc_model = self.sapm
            elif model == 'singlediode':
                self._dc_model = self.singlediode
            elif model == 'pvwatts':
                self._dc_model = self.pvwatts_dc
            else:
                raise ValueError(model + ' is not a valid DC power model')
        else:
            self._dc_model = partial(model, self)

    def infer_dc_model(self):
        params = set(self.system.module_parameters.keys())
        if set(['A0', 'A1', 'C7']) <= params:
            return self.sapm
        elif set(['a_ref', 'I_L_ref', 'I_o_ref', 'R_sh_ref', 'R_s']) <= params:
            return self.singlediode
        elif set(['pdc0', 'gamma_pdc']) <= params:
            return self.pvwatts_dc
        else:
            raise ValueError('could not infer DC model from '
                             'system.module_parameters')

    def sapm(self):
        self.dc = self.system.sapm(self.effective_irradiance/1000.,
                                   self.temps['temp_cell'])

        self.dc = self.system.scale_voltage_current_power(self.dc)

        return self

    def singlediode(self):
        (photocurrent, saturation_current, resistance_series,
         resistance_shunt, nNsVth) = (
            self.system.calcparams_desoto(self.effective_irradiance,
                                          self.temps['temp_cell']))

        self.desoto = (photocurrent, saturation_current, resistance_series,
                       resistance_shunt, nNsVth)

        self.dc = self.system.singlediode(
            photocurrent, saturation_current, resistance_series,
            resistance_shunt, nNsVth)

        self.dc = self.system.scale_voltage_current_power(self.dc).fillna(0)

        return self

    def pvwatts_dc(self):
        self.dc = self.system.pvwatts_dc(self.effective_irradiance,
                                         self.temps['temp_cell'])
        return self

    @property
    def ac_model(self):
        return self._ac_model

    @ac_model.setter
    def ac_model(self, model):
        if model is None:
            self._ac_model = self.infer_ac_model()
        elif isinstance(model, str):
            model = model.lower()
            if model == 'snlinverter':
                self._ac_model = self.snlinverter
            elif model == 'adrinverter':
                self._ac_model = self.adrinverter
            elif model == 'pvwatts':
                self._ac_model = self.pvwatts_inverter
            else:
                raise ValueError(model + ' is not a valid AC power model')
        else:
            self._ac_model = partial(model, self)

    def infer_ac_model(self):
        inverter_params = set(self.system.inverter_parameters.keys())
        module_params = set(self.system.module_parameters.keys())
        if set(['C0', 'C1', 'C2']) <= inverter_params:
            return self.snlinverter
        elif set(['ADRCoefficients']) <= inverter_params:
            return self.adrinverter
        elif set(['pdc0']) <= module_params:
            return self.pvwatts_inverter
        else:
            raise ValueError('could not infer AC model from '
                             'system.inverter_parameters')

    def snlinverter(self):
        self.ac = self.system.snlinverter(self.dc['v_mp'], self.dc['p_mp'])
        return self

    def adrinverter(self):
        self.ac = self.system.adrinverter(self.dc['v_mp'], self.dc['p_mp'])
        return self

    def pvwatts_inverter(self):
        self.ac = self.system.pvwatts_ac(self.dc).fillna(0)
        return self

    @property
    def aoi_model(self):
        return self._aoi_model

    @aoi_model.setter
    def aoi_model(self, model):
        if model is None:
            self._aoi_model = self.infer_aoi_model()
        elif isinstance(model, str):
            model = model.lower()
            if model == 'ashrae':
                self._aoi_model = self.ashrae_aoi_loss
            elif model == 'physical':
                self._aoi_model = self.physical_aoi_loss
            elif model == 'sapm':
                self._aoi_model = self.sapm_aoi_loss
            elif model == 'no_loss':
                self._aoi_model = self.no_aoi_loss
            else:
                raise ValueError(model + ' is not a valid aoi loss model')
        else:
            self._aoi_model = partial(model, self)

    def infer_aoi_model(self):
        params = set(self.system.module_parameters.keys())
        if set(['K', 'L', 'n']) <= params:
            return self.physical_aoi_loss
        elif set(['B5', 'B4', 'B3', 'B2', 'B1', 'B0']) <= params:
            return self.sapm_aoi_loss
        elif set(['b']) <= params:
            return self.ashrae_aoi_loss
        else:
            raise ValueError('could not infer AOI model from '
                             'system.module_parameters')

    def ashrae_aoi_loss(self):
        self.aoi_modifier = self.system.ashraeiam(self.aoi)
        return self

    def physical_aoi_loss(self):
        self.aoi_modifier = self.system.physicaliam(self.aoi)
        return self

    def sapm_aoi_loss(self):
        self.aoi_modifier = self.system.sapm_aoi_loss(self.aoi)
        return self

    def no_aoi_loss(self):
        self.aoi_modifier = 1.0
        return self

    @property
    def spectral_model(self):
        return self._spectral_model

    @spectral_model.setter
    def spectral_model(self, model):
        if model is None:
            self._spectral_model = self.infer_spectral_model()
        elif isinstance(model, str):
            model = model.lower()
            if model == 'first_solar':
                self._spectral_model = self.first_solar_spectral_loss
            elif model == 'sapm':
                self._spectral_model = self.sapm_spectral_loss
            elif model == 'no_loss':
                self._spectral_model = self.no_spectral_loss
            else:
                raise ValueError(model + ' is not a valid spectral loss model')
        else:
            self._spectral_model = partial(model, self)

    def infer_spectral_model(self):
        params = set(self.system.module_parameters.keys())
        if set(['A4', 'A3', 'A2', 'A1', 'A0']) <= params:
            return self.sapm_spectral_loss
        elif ((('Technology' in params or
                'Material' in params) and
               (pvsystem._infer_cell_type() is not None)) or
              'first_solar_spectral_coefficients' in params):
            return self.first_solar_spectral_loss
        else:
            raise ValueError('could not infer spectral model from '
                             'system.module_parameters. Check that the '
                             'parameters contain valid '
                             'first_solar_spectral_coefficients or a valid '
                             'Material or Technology value')

    def first_solar_spectral_loss(self):
        self.spectral_modifier = self.system.first_solar_spectral_loss(
                                        self.weather['precipitable_water'],
                                        self.airmass['airmass_absolute'])
        return self

    def sapm_spectral_loss(self):
        self.spectral_modifier = self.system.sapm_spectral_loss(
            self.airmass['airmass_absolute'])
        return self

    def no_spectral_loss(self):
        self.spectral_modifier = 1
        return self

    @property
    def temp_model(self):
        return self._temp_model

    @temp_model.setter
    def temp_model(self, model):
        if model is None:
            self._temp_model = self.infer_temp_model()
        elif isinstance(model, str):
            model = model.lower()
            if model == 'sapm':
                self._temp_model = self.sapm_temp
            else:
                raise ValueError(model + ' is not a valid temp model')
        else:
            self._temp_model = partial(model, self)

    def infer_temp_model(self):
        raise NotImplementedError

    def sapm_temp(self):
        self.temps = self.system.sapm_celltemp(self.total_irrad['poa_global'],
                                               self.weather['wind_speed'],
                                               self.weather['temp_air'])
        return self

    @property
    def losses_model(self):
        return self._losses_model

    @losses_model.setter
    def losses_model(self, model):
        if model is None:
            self._losses_model = self.infer_losses_model()
        elif isinstance(model, str):
            model = model.lower()
            if model == 'pvwatts':
                self._losses_model = self.pvwatts_losses
            elif model == 'no_loss':
                self._losses_model = self.no_extra_losses
            else:
                raise ValueError(model + ' is not a valid losses model')
        else:
            self._losses_model = partial(model, self)





class ForecastModel(object):
    access_url_key = 'NetcdfSubset'
    catalog_url = 'http://thredds.ucar.edu/thredds/catalog.xml'
    base_tds_url = catalog_url.split('/thredds/')[0]
    data_format = 'netcdf'
    vert_level = 100000

    units = {
        'temp_air': 'C',
        'wind_speed': 'm/s',
        'ghi': 'W/m^2',
        'ghi_raw': 'W/m^2',
        'dni': 'W/m^2',
        'dhi': 'W/m^2',
        'total_clouds': '%',
        'low_clouds': '%',
        'mid_clouds': '%',
        'high_clouds': '%'}

    def __init__(self, model_type, model_name, set_type):
        self.model_type = model_type
        self.model_name = model_name
        self.set_type = set_type
        self.connected = False

    def connect_to_catalog(self):
        self.catalog = TDSCatalog(self.catalog_url)
        self.fm_models = TDSCatalog(
            self.catalog.catalog_refs[self.model_type].href)
        self.fm_models_list = sorted(list(self.fm_models.catalog_refs.keys()))

        try:
            model_url = self.fm_models.catalog_refs[self.model_name].href
        except ParseError:
            raise ParseError(self.model_name + ' model may be unavailable.')

        try:
            self.model = TDSCatalog(model_url)
        except HTTPError:
            try:
                self.model = TDSCatalog(model_url)
            except HTTPError:
                raise HTTPError(self.model_name + ' model may be unavailable.')

        self.datasets_list = list(self.model.datasets.keys())
        self.set_dataset()
        self.connected = True

    def __repr__(self):
        return '{}, {}'.format(self.model_name, self.set_type)

    def set_dataset(self):
        '''
        Retrieves the designated dataset, creates NCSS object, and
        creates a NCSS query object.
        '''

        keys = list(self.model.datasets.keys())
        labels = [item.split()[0].lower() for item in keys]
        if self.set_type == 'best':
            self.dataset = self.model.datasets[keys[labels.index('best')]]
        elif self.set_type == 'latest':
            self.dataset = self.model.datasets[keys[labels.index('latest')]]
        elif self.set_type == 'full':
            self.dataset = self.model.datasets[keys[labels.index('full')]]

        self.access_url = self.dataset.access_urls[self.access_url_key]
        self.ncss = NCSS(self.access_url)
        self.query = self.ncss.query()

    def set_query_latlon(self):
        '''
        Sets the NCSS query location latitude and longitude.
        '''

        if (isinstance(self.longitude, list) and
                isinstance(self.latitude, list)):
            self.lbox = True
            # west, east, south, north
            self.query.lonlat_box(self.latitude[0], self.latitude[1],
                                  self.longitude[0], self.longitude[1])
        else:
            self.lbox = False
            self.query.lonlat_point(self.longitude, self.latitude)

    def set_location(self, time, latitude, longitude):
        '''
        Sets the location for the query.

        Parameters
        ----------
        time: datetime or DatetimeIndex
            Time range of the query.
        '''
        if isinstance(time, datetime.datetime):
            tzinfo = time.tzinfo
        else:
            tzinfo = time.tz

        if tzinfo is None:
            self.location = Location(latitude, longitude)
        else:
            self.location = Location(latitude, longitude, tz=tzinfo)

    def get_data(self, latitude, longitude, start, end,
                 vert_level=None, query_variables=None,
                 close_netcdf_data=True):
        if not self.connected:
            self.connect_to_catalog()

        if vert_level is not None:
            self.vert_level = vert_level

        if query_variables is None:
            self.query_variables = list(self.variables.values())
        else:
            self.query_variables = query_variables

        self.latitude = latitude
        self.longitude = longitude
        self.set_query_latlon()  # modifies self.query
        self.set_location(start, latitude, longitude)

        self.start = start
        self.end = end
        self.query.time_range(self.start, self.end)

        self.query.vertical_level(self.vert_level)
        self.query.variables(*self.query_variables)
        self.query.accept(self.data_format)

        self.netcdf_data = self.ncss.get_data(self.query)

        # might be better to go to xarray here so that we can handle
        # higher dimensional data for more advanced applications
        self.data = self._netcdf2pandas(self.netcdf_data, self.query_variables)

        if close_netcdf_data:
            self.netcdf_data.close()

        return self.data

    def process_data(self, data, **kwargs):
       
        data = self.rename(data)
        return data

    def get_processed_data(self, *args, **kwargs):
       
        return self.process_data(self.get_data(*args, **kwargs), **kwargs)

    def rename(self, data, variables=None):
      
        if variables is None:
            variables = self.variables
        return data.rename(columns={y: x for x, y in variables.items()})

    def _netcdf2pandas(self, netcdf_data, query_variables):
        """
        Transforms data from netcdf to pandas DataFrame.

        Parameters
        ----------
        data: netcdf
            Data returned from UNIDATA NCSS query.
        query_variables: list
            The variables requested.

        Returns
        -------
        pd.DataFrame
        """
        # set self.time
        try:
            time_var = 'time'
            self.set_time(netcdf_data.variables[time_var])
        except KeyError:
            # which model does this dumb thing?
            time_var = 'time1'
            self.set_time(netcdf_data.variables[time_var])

        data_dict = {key: data[:].squeeze() for key, data in
                     netcdf_data.variables.items() if key in query_variables}

        return pd.DataFrame(data_dict, index=self.time)

    def set_time(self, time):
        times = num2date(time[:].squeeze(), time.units)
        self.time = pd.DatetimeIndex(pd.Series(times), tz=self.location.tz)

    def cloud_cover_to_ghi_linear(self, cloud_cover, ghi_clear, offset=35,
                                  **kwargs):

        offset = offset / 100.
        cloud_cover = cloud_cover / 100.
        ghi = (offset + (1 - offset) * (1 - cloud_cover)) * ghi_clear
        return ghi

    def cloud_cover_to_irradiance_clearsky_scaling(self, cloud_cover,
                                                   method='linear',
                                                   **kwargs):

        solpos = self.location.get_solarposition(cloud_cover.index)
        cs = self.location.get_clearsky(cloud_cover.index, model='ineichen',
                                        solar_position=solpos)

        method = method.lower()
        if method == 'linear':
            ghi = self.cloud_cover_to_ghi_linear(cloud_cover, cs['ghi'],
                                                 **kwargs)
        else:
            raise ValueError('invalid method argument')

        dni = disc(ghi, solpos['zenith'], cloud_cover.index)['dni']
        dhi = ghi - dni * np.cos(np.radians(solpos['zenith']))

        irrads = pd.DataFrame({'ghi': ghi, 'dni': dni, 'dhi': dhi}).fillna(0)
        return irrads

    def cloud_cover_to_transmittance_linear(self, cloud_cover, offset=0.75,
                                            **kwargs):

        transmittance = ((100.0 - cloud_cover) / 100.0) * offset

        return transmittance

    def cloud_cover_to_irradiance_liujordan(self, cloud_cover, **kwargs):
        
        # in principle, get_solarposition could use the forecast
        # pressure, temp, etc., but the cloud cover forecast is not
        # accurate enough to justify using these minor corrections
        solar_position = self.location.get_solarposition(cloud_cover.index)
        dni_extra = extraradiation(cloud_cover.index)
        airmass = self.location.get_airmass(cloud_cover.index)

        transmittance = self.cloud_cover_to_transmittance_linear(cloud_cover,
                                                                 **kwargs)

        irrads = liujordan(solar_position['apparent_zenith'],
                           transmittance, airmass['airmass_absolute'],
                           dni_extra=dni_extra)
        irrads = irrads.fillna(0)

        return irrads

    def cloud_cover_to_irradiance(self, cloud_cover, how='clearsky_scaling',
                                  **kwargs):
        """
        Convert cloud cover to irradiance. A wrapper method.

        Parameters
        ----------
        cloud_cover : Series
        how : str, default 'clearsky_scaling'
            Selects the method for conversion. Can be one of
            clearsky_scaling or liujordan.
        **kwargs
            Passed to the selected method.

        Returns
        -------
        irradiance : DataFrame
            Columns include ghi, dni, dhi
        """

        how = how.lower()
        if how == 'clearsky_scaling':
            irrads = self.cloud_cover_to_irradiance_clearsky_scaling(
                cloud_cover, **kwargs)
        elif how == 'liujordan':
            irrads = self.cloud_cover_to_irradiance_liujordan(
                cloud_cover, **kwargs)
        else:
            raise ValueError('invalid how argument')

        return irrads

    def kelvin_to_celsius(self, temperature):
              return temperature - 273.15

    def isobaric_to_ambient_temperature(self, data):
       

        P = data['pressure'] / 100.0
        Tiso = data['temperature_iso']
        Td = data['temperature_dew_iso'] - 273.15

        # saturation water vapor pressure
        e = 6.11 * 10**((7.5 * Td) / (Td + 273.3))

        # saturation water vapor mixing ratio
        w = 0.622 * (e / (P - e))

        T = Tiso - ((2.501 * 10.**6) / 1005.7) * w

        return T

    def uv_to_speed(self, data):
     
        wind_speed = np.sqrt(data['wind_speed_u']**2 + data['wind_speed_v']**2)

        return wind_speed

    def gust_to_speed(self, data, scaling=1/1.4):
       
        wind_speed = data['wind_speed_gust'] * scaling

        return wind_speed


class GFS(ForecastModel):


    _resolutions = ['Half', 'Quarter']

    def __init__(self, resolution='half', set_type='best'):
        model_type = 'Forecast Model Data'

        resolution = resolution.title()
        if resolution not in self._resolutions:
            raise ValueError('resolution must in {}'.format(self._resolutions))

        model = 'GFS {} Degree Forecast'.format(resolution)

        self.variables = {
            'temp_air': 'Temperature_surface',
            'wind_speed_gust': 'Wind_speed_gust_surface',
            'wind_speed_u': 'u-component_of_wind_isobaric',
            'wind_speed_v': 'v-component_of_wind_isobaric',
            'total_clouds': 'Total_cloud_cover_entire_atmosphere_Mixed_intervals_Average',
            'low_clouds': 'Total_cloud_cover_low_cloud_Mixed_intervals_Average',
            'mid_clouds': 'Total_cloud_cover_middle_cloud_Mixed_intervals_Average',
            'high_clouds': 'Total_cloud_cover_high_cloud_Mixed_intervals_Average',
            'boundary_clouds': 'Total_cloud_cover_boundary_layer_cloud_Mixed_intervals_Average',
            'convect_clouds': 'Total_cloud_cover_convective_cloud',
            'ghi_raw': 'Downward_Short-Wave_Radiation_Flux_surface_Mixed_intervals_Average', }

        self.output_variables = [
            'temp_air',
            'wind_speed',
            'ghi',
            'dni',
            'dhi',
            'total_clouds',
            'low_clouds',
            'mid_clouds',
            'high_clouds']

        super(GFS, self).__init__(model_type, model, set_type)

    def process_data(self, data, cloud_cover='total_clouds', **kwargs):
  
        data = super(GFS, self).process_data(data, **kwargs)
        data['temp_air'] = self.kelvin_to_celsius(data['temp_air'])
        data['wind_speed'] = self.uv_to_speed(data)
        irrads = self.cloud_cover_to_irradiance(data[cloud_cover], **kwargs)
        data = data.join(irrads, how='outer')
        return data[self.output_variables]


class HRRR_ESRL(ForecastModel):
 

    def __init__(self, set_type='best'):
        import warnings
        warnings.warn('HRRR_ESRL is an experimental model and is not always available.')

        model_type = 'Forecast Model Data'
        model = 'GSD HRRR CONUS 3km surface'

        self.variables = {
            'temp_air': 'Temperature_surface',
            'wind_speed_gust': 'Wind_speed_gust_surface',
            'total_clouds': 'Total_cloud_cover_entire_atmosphere',
            'low_clouds': 'Low_cloud_cover_UnknownLevelType-214',
            'mid_clouds': 'Medium_cloud_cover_UnknownLevelType-224',
            'high_clouds': 'High_cloud_cover_UnknownLevelType-234',
            'ghi_raw': 'Downward_short-wave_radiation_flux_surface', }

        self.output_variables = [
            'temp_air',
            'wind_speed'
            'ghi_raw',
            'ghi',
            'dni',
            'dhi',
            'total_clouds',
            'low_clouds',
            'mid_clouds',
            'high_clouds']

        super(HRRR_ESRL, self).__init__(model_type, model, set_type)

    def process_data(self, data, cloud_cover='total_clouds', **kwargs):
    
        data = super(HRRR_ESRL, self).process_data(data, **kwargs)
        data['temp_air'] = self.kelvin_to_celsius(data['temp_air'])
        data['wind_speed'] = self.gust_to_speed(data)
        irrads = self.cloud_cover_to_irradiance(data[cloud_cover], **kwargs)
        data = data.join(irrads, how='outer')
        return data[self.output_variables]


class NAM(ForecastModel):
 
    def __init__(self, set_type='best'):
        model_type = 'Forecast Model Data'
        model = 'NAM CONUS 12km from CONDUIT'

        self.variables = {
            'temp_air': 'Temperature_surface',
            'wind_speed_gust': 'Wind_speed_gust_surface',
            'total_clouds': 'Total_cloud_cover_entire_atmosphere_single_layer',
            'low_clouds': 'Low_cloud_cover_low_cloud',
            'mid_clouds': 'Medium_cloud_cover_middle_cloud',
            'high_clouds': 'High_cloud_cover_high_cloud',
            'ghi_raw': 'Downward_Short-Wave_Radiation_Flux_surface', }

        self.output_variables = [
            'temp_air',
            'wind_speed',
            'ghi',
            'dni',
            'dhi',
            'total_clouds',
            'low_clouds',
            'mid_clouds',
            'high_clouds']

        super(NAM, self).__init__(model_type, model, set_type)

    def process_data(self, data, cloud_cover='total_clouds', **kwargs):
       

        data = super(NAM, self).process_data(data, **kwargs)
        data['temp_air'] = self.kelvin_to_celsius(data['temp_air'])
        data['wind_speed'] = self.gust_to_speed(data)
        irrads = self.cloud_cover_to_irradiance(data[cloud_cover], **kwargs)
        data = data.join(irrads, how='outer')
        return data[self.output_variables]


class HRRR(ForecastModel):
   
    def __init__(self, set_type='best'):
        model_type = 'Forecast Model Data'
        model = 'NCEP HRRR CONUS 2.5km'

        self.variables = {
            'temperature_dew_iso': 'Dewpoint_temperature_isobaric',
            'temperature_iso': 'Temperature_isobaric',
            'pressure': 'Pressure_surface',
            'wind_speed_gust': 'Wind_speed_gust_surface',
            'total_clouds': 'Total_cloud_cover_entire_atmosphere',
            'low_clouds': 'Low_cloud_cover_low_cloud',
            'mid_clouds': 'Medium_cloud_cover_middle_cloud',
            'high_clouds': 'High_cloud_cover_high_cloud',
            'condensation_height': 'Geopotential_height_adiabatic_condensation_lifted'}

        self.output_variables = [
            'temp_air',
            'wind_speed',
            'ghi',
            'dni',
            'dhi',
            'total_clouds',
            'low_clouds',
            'mid_clouds',
            'high_clouds', ]

        super(HRRR, self).__init__(model_type, model, set_type)

    def process_data(self, data, cloud_cover='total_clouds', **kwargs):
    

        data = super(HRRR, self).process_data(data, **kwargs)
        data['temp_air'] = self.isobaric_to_ambient_temperature(data)
        data['temp_air'] = self.kelvin_to_celsius(data['temp_air'])
        data['wind_speed'] = self.gust_to_speed(data)
        irrads = self.cloud_cover_to_irradiance(data[cloud_cover], **kwargs)
        data = data.join(irrads, how='outer')
        return data[self.output_variables]


class NDFD(ForecastModel):
    
    def __init__(self, set_type='best'):
        model_type = 'Forecast Products and Analyses'
        model = 'National Weather Service CONUS Forecast Grids (CONDUIT)'
        self.variables = {
            'temp_air': 'Temperature_surface',
            'wind_speed': 'Wind_speed_surface',
            'wind_speed_gust': 'Wind_speed_gust_surface',
            'total_clouds': 'Total_cloud_cover_surface', }
        self.output_variables = [
            'temp_air',
            'wind_speed',
            'ghi',
            'dni',
            'dhi',
            'total_clouds', ]
        super(NDFD, self).__init__(model_type, model, set_type)

    def process_data(self, data, **kwargs):
        """
        Defines the steps needed to convert raw forecast data
        into processed forecast data.

        Parameters
        ----------
        data: DataFrame
            Raw forecast data

        Returns
        -------
        data: DataFrame
            Processed forecast data.
        """

        cloud_cover = 'total_clouds'
        data = super(NDFD, self).process_data(data, **kwargs)
        data['temp_air'] = self.kelvin_to_celsius(data['temp_air'])
        irrads = self.cloud_cover_to_irradiance(data[cloud_cover], **kwargs)
        data = data.join(irrads, how='outer')
        return data[self.output_variables]


class RAP(ForecastModel):
 

    _resolutions = ['20', '40']

    def __init__(self, resolution='20', set_type='best'):

        resolution = str(resolution)
        if resolution not in self._resolutions:
            raise ValueError('resolution must in {}'.format(self._resolutions))

        model_type = 'Forecast Model Data'
        model = 'Rapid Refresh CONUS {}km'.format(resolution)
        self.variables = {
            'temp_air': 'Temperature_surface',
            'wind_speed_gust': 'Wind_speed_gust_surface',
            'total_clouds': 'Total_cloud_cover_entire_atmosphere',
            'low_clouds': 'Low_cloud_cover_low_cloud',
            'mid_clouds': 'Medium_cloud_cover_middle_cloud',
            'high_clouds': 'High_cloud_cover_high_cloud', }
        self.output_variables = [
            'temp_air',
            'wind_speed',
            'ghi',
            'dni',
            'dhi',
            'total_clouds',
            'low_clouds',
            'mid_clouds',
            'high_clouds', ]
        super(RAP, self).__init__(model_type, model, set_type)

    def process_data(self, data, cloud_cover='total_clouds', **kwargs):
       

        data = super(RAP, self).process_data(data, **kwargs)
        data['temp_air'] = self.kelvin_to_celsius(data['temp_air'])
        data['wind_speed'] = self.gust_to_speed(data)
        irrads = self.cloud_cover_to_irradiance(data[cloud_cover], **kwargs)
        data = data.join(irrads, how='outer')
        return data[self.output_variables]

    def infer_losses_model(self):
        raise NotImplementedError

    def pvwatts_losses(self):
        self.losses = (100 - self.system.pvwatts_losses()) / 100.
        self.ac *= self.losses
        return self

    def no_extra_losses(self):
        self.losses = 1
        return self

    def effective_irradiance_model(self):
        fd = self.system.module_parameters.get('FD', 1.)
        self.effective_irradiance = self.spectral_modifier * (
            self.total_irrad['poa_direct']*self.aoi_modifier +
            fd*self.total_irrad['poa_diffuse'])
        return self

    def complete_irradiance(self, times=None, weather=None):
        if weather is not None:
            self.weather = weather
        if times is not None:
            self.times = times
        self.solar_position = self.location.get_solarposition(
            self.times, method=self.solar_position_method)
        icolumns = set(self.weather.columns)
        wrn_txt = ("This function is not safe at the moment.\n" +
                   "Results can be too high or negative.\n" +
                   "Help to improve this function on github:\n" +
                   "https://github.com/pvlib/pvlib-python \n")

        if {'ghi', 'dhi'} <= icolumns and 'dni' not in icolumns:
            clearsky = self.location.get_clearsky(
                times, solar_position=self.solar_position)
            self.weather.loc[:, 'dni'] = pvlib.irradiance.dni(
                self.weather.loc[:, 'ghi'], self.weather.loc[:, 'dhi'],
                self.solar_position.zenith,
                clearsky_dni=clearsky['dni'],
                clearsky_tolerance=1.1)
        elif {'dni', 'dhi'} <= icolumns and 'ghi' not in icolumns:
            warnings.warn(wrn_txt, UserWarning)
            self.weather.loc[:, 'ghi'] = (
                self.weather.dni * tools.cosd(self.solar_position.zenith) +
                self.weather.dhi)
        elif {'dni', 'ghi'} <= icolumns and 'dhi' not in icolumns:
            warnings.warn(wrn_txt, UserWarning)
            self.weather.loc[:, 'dhi'] = (
                self.weather.ghi - self.weather.dni *
                tools.cosd(self.solar_position.zenith))

        return self

    def prepare_inputs(self, times=None, weather=None):
       
        if weather is not None:
            self.weather = weather
        if self.weather is None:
            self.weather = pd.DataFrame()

        if times is not None:
            self.times = times

        self.solar_position = self.location.get_solarposition(
            self.times, method=self.solar_position_method)

        self.airmass = self.location.get_airmass(
            solar_position=self.solar_position, model=self.airmass_model)

        if not any([x in ['ghi', 'dni', 'dhi'] for x in self.weather.columns]):
            self.weather[['ghi', 'dni', 'dhi']] = self.location.get_clearsky(
                self.solar_position.index, self.clearsky_model,
                solar_position=self.solar_position,
                airmass_absolute=self.airmass['airmass_absolute'])

        if not {'ghi', 'dni', 'dhi'} <= set(self.weather.columns):
            raise ValueError(
                "Uncompleted irradiance data set. Please check you input " +
                "data.\nData set needs to have 'dni', 'dhi' and 'ghi'.\n" +
                "Detected data: {0}".format(list(self.weather.columns)))

        # PVSystem.get_irradiance and SingleAxisTracker.get_irradiance
        # and PVSystem.get_aoi and SingleAxisTracker.get_aoi
        # have different method signatures. Use partial to handle
        # the differences.
        if isinstance(self.system, SingleAxisTracker):
            self.tracking = self.system.singleaxis(
                self.solar_position['apparent_zenith'],
                self.solar_position['azimuth'])
            self.tracking['surface_tilt'] = (
                self.tracking['surface_tilt']
                    .fillna(self.system.axis_tilt))
            self.tracking['surface_azimuth'] = (
                self.tracking['surface_azimuth']
                    .fillna(self.system.axis_azimuth))
            self.aoi = self.tracking['aoi']
            get_irradiance = partial(
                self.system.get_irradiance,
                self.tracking['surface_tilt'],
                self.tracking['surface_azimuth'],
                self.solar_position['apparent_zenith'],
                self.solar_position['azimuth'])
        else:
            self.aoi = self.system.get_aoi(
                self.solar_position['apparent_zenith'],
                self.solar_position['azimuth'])
            get_irradiance = partial(
                self.system.get_irradiance,
                self.solar_position['apparent_zenith'],
                self.solar_position['azimuth'])

        self.total_irrad = get_irradiance(
            self.weather['dni'],
            self.weather['ghi'],
            self.weather['dhi'],
            airmass=self.airmass['airmass_relative'],
            model=self.transposition_model)

        if self.weather.get('wind_speed') is None:
            self.weather['wind_speed'] = 0
        if self.weather.get('temp_air') is None:
            self.weather['temp_air'] = 20
        return self

    def run_model(self, times=None, weather=None):
       

        self.prepare_inputs(times, weather)
        self.aoi_model()
        self.spectral_model()
        self.effective_irradiance_model()
        self.temp_model()
        self.dc_model()
        self.ac_model()
        self.losses_model()

        return self

        try:
                from importlib import reload
except ImportError:
    try:
        from imp import reload
    except ImportError:
        pass

import numpy as np
import pandas as pd

from pvlib import atmosphere
from pvlib.tools import datetime_to_djd, djd_to_datetime


def get_solarposition(time, latitude, longitude,
                      altitude=None, pressure=None,
                      method='nrel_numpy',
                      temperature=12, **kwargs):
   

    if altitude is None and pressure is None:
        altitude = 0.
        pressure = 101325.
    elif altitude is None:
        altitude = atmosphere.pres2alt(pressure)
    elif pressure is None:
        pressure = atmosphere.alt2pres(altitude)

    method = method.lower()
    if isinstance(time, dt.datetime):
        time = pd.DatetimeIndex([time, ])

    if method == 'nrel_c':
        ephem_df = spa_c(time, latitude, longitude, pressure, temperature,
                         **kwargs)
    elif method == 'nrel_numba':
        ephem_df = spa_python(time, latitude, longitude, altitude,
                              pressure, temperature,
                              how='numba', **kwargs)
    elif method == 'nrel_numpy':
        ephem_df = spa_python(time, latitude, longitude, altitude,
                              pressure, temperature,
                              how='numpy', **kwargs)
    elif method == 'pyephem':
        ephem_df = pyephem(time, latitude, longitude,
                           altitude=altitude,
                           pressure=pressure,
                           temperature=temperature, **kwargs)
    elif method == 'ephemeris':
        ephem_df = ephemeris(time, latitude, longitude, pressure, temperature,
                             **kwargs)
    else:
        raise ValueError('Invalid solar position method')

    return ephem_df


def spa_c(time, latitude, longitude, pressure=101325, altitude=0,
          temperature=12, delta_t=67.0,
          raw_spa_output=False):
    try:
        from pvlib.spa_c_files.spa_py import spa_calc
    except ImportError:
        raise ImportError('Could not import built-in SPA calculator. ' +
                          'You may need to recompile the SPA code.')

    # if localized, convert to UTC. otherwise, assume UTC.
    try:
        time_utc = time.tz_convert('UTC')
    except TypeError:
        time_utc = time

    spa_out = []

    for date in time_utc:
        spa_out.append(spa_calc(year=date.year,
                                month=date.month,
                                day=date.day,
                                hour=date.hour,
                                minute=date.minute,
                                second=date.second,
                                timezone=0,  # date uses utc time
                                latitude=latitude,
                                longitude=longitude,
                                elevation=altitude,
                                pressure=pressure / 100,
                                temperature=temperature,
                                delta_t=delta_t
                                ))

    spa_df = pd.DataFrame(spa_out, index=time)

    if raw_spa_output:
        return spa_df
    else:
        dfout = pd.DataFrame({'azimuth': spa_df['azimuth'],
                              'apparent_zenith': spa_df['zenith'],
                              'apparent_elevation': spa_df['e'],
                              'elevation': spa_df['e0'],
                              'zenith': 90 - spa_df['e0']})

        return dfout


def _spa_python_import(how):
    """Compile spa.py appropriately"""

    from pvlib import spa

    # check to see if the spa module was compiled with numba
    using_numba = spa.USE_NUMBA

    if how == 'numpy' and using_numba:
        # the spa module was compiled to numba code, so we need to
        # reload the module without compiling
        # the PVLIB_USE_NUMBA env variable is used to tell the module
        # to not compile with numba
        os.environ['PVLIB_USE_NUMBA'] = '0'
        spa = reload(spa)
        del os.environ['PVLIB_USE_NUMBA']
    elif how == 'numba' and not using_numba:
        # The spa module was not compiled to numba code, so set
        # PVLIB_USE_NUMBA so it does compile to numba on reload.
        os.environ['PVLIB_USE_NUMBA'] = '1'
        spa = reload(spa)
        del os.environ['PVLIB_USE_NUMBA']
    elif how != 'numba' and how != 'numpy':
        raise ValueError("how must be either 'numba' or 'numpy'")

    return spa


def spa_python(time, latitude, longitude,
               altitude=0, pressure=101325, temperature=12, delta_t=67.0,
               atmos_refract=None, how='numpy', numthreads=4, **kwargs):
    lat = latitude
    lon = longitude
    elev = altitude
    pressure = pressure / 100  # pressure must be in millibars for calculation

    atmos_refract = atmos_refract or 0.5667

    if not isinstance(time, pd.DatetimeIndex):
        try:
            time = pd.DatetimeIndex(time)
        except (TypeError, ValueError):
            time = pd.DatetimeIndex([time, ])

    unixtime = np.array(time.astype(np.int64)/10**9)

    spa = _spa_python_import(how)

    delta_t = delta_t or spa.calculate_deltat(time.year, time.month)

    app_zenith, zenith, app_elevation, elevation, azimuth, eot = \
        spa.solar_position(unixtime, lat, lon, elev, pressure, temperature,
                           delta_t, atmos_refract, numthreads)

    result = pd.DataFrame({'apparent_zenith': app_zenith, 'zenith': zenith,
                           'apparent_elevation': app_elevation,
                           'elevation': elevation, 'azimuth': azimuth,
                           'equation_of_time': eot},
                          index=time)

    return result


def get_sun_rise_set_transit(time, latitude, longitude, how='numpy',
                             delta_t=67.0,
                             numthreads=4):
 
    lat = latitude
    lon = longitude

    if not isinstance(time, pd.DatetimeIndex):
        try:
            time = pd.DatetimeIndex(time)
        except (TypeError, ValueError):
            time = pd.DatetimeIndex([time, ])

    # must convert to midnight UTC on day of interest
    utcday = pd.DatetimeIndex(time.date).tz_localize('UTC')
    unixtime = np.array(utcday.astype(np.int64)/10**9)

    spa = _spa_python_import(how)

    delta_t = delta_t or spa.calculate_deltat(time.year, time.month)

    transit, sunrise, sunset = spa.transit_sunrise_sunset(
        unixtime, lat, lon, delta_t, numthreads)

    # arrays are in seconds since epoch format, need to conver to timestamps
    transit = pd.to_datetime(transit*1e9, unit='ns', utc=True).tz_convert(
        time.tz).tolist()
    sunrise = pd.to_datetime(sunrise*1e9, unit='ns', utc=True).tz_convert(
        time.tz).tolist()
    sunset = pd.to_datetime(sunset*1e9, unit='ns', utc=True).tz_convert(
        time.tz).tolist()

    result = pd.DataFrame({'transit': transit,
                           'sunrise': sunrise,
                           'sunset': sunset}, index=time)

    return result


def _ephem_setup(latitude, longitude, altitude, pressure, temperature):
    import ephem
    # initialize a PyEphem observer
    obs = ephem.Observer()
    obs.lat = str(latitude)
    obs.lon = str(longitude)
    obs.elevation = altitude
    obs.pressure = pressure / 100.  # convert to mBar
    obs.temp = temperature

    # the PyEphem sun
    sun = ephem.Sun()
    return obs, sun


def pyephem(time, latitude, longitude, altitude=0, pressure=101325,
            temperature=12):
 
   
    try:
        import ephem
    except ImportError:
        raise ImportError('PyEphem must be installed')

    # if localized, convert to UTC. otherwise, assume UTC.
    try:
        time_utc = time.tz_convert('UTC')
    except TypeError:
        time_utc = time

    sun_coords = pd.DataFrame(index=time)

    obs, sun = _ephem_setup(latitude, longitude, altitude,
                            pressure, temperature)

    # make and fill lists of the sun's altitude and azimuth
    # this is the pressure and temperature corrected apparent alt/az.
    alts = []
    azis = []
    for thetime in time_utc:
        obs.date = ephem.Date(thetime)
        sun.compute(obs)
        alts.append(sun.alt)
        azis.append(sun.az)

    sun_coords['apparent_elevation'] = alts
    sun_coords['apparent_azimuth'] = azis

    # redo it for p=0 to get no atmosphere alt/az
    obs.pressure = 0
    alts = []
    azis = []
    for thetime in time_utc:
        obs.date = ephem.Date(thetime)
        sun.compute(obs)
        alts.append(sun.alt)
        azis.append(sun.az)

    sun_coords['elevation'] = alts
    sun_coords['azimuth'] = azis

    # convert to degrees. add zenith
    sun_coords = np.rad2deg(sun_coords)
    sun_coords['apparent_zenith'] = 90 - sun_coords['apparent_elevation']
    sun_coords['zenith'] = 90 - sun_coords['elevation']

    return sun_coords


def ephemeris(time, latitude, longitude, pressure=101325, temperature=12):

    Latitude = latitude
    Longitude = -1 * longitude

    Abber = 20 / 3600.
    LatR = np.radians(Latitude)

    # the SPA algorithm needs time to be expressed in terms of
    # decimal UTC hours of the day of the year.

    # if localized, convert to UTC. otherwise, assume UTC.
    try:
        time_utc = time.tz_convert('UTC')
    except TypeError:
        time_utc = time

    # strip out the day of the year and calculate the decimal hour
    DayOfYear = time_utc.dayofyear
    DecHours = (time_utc.hour + time_utc.minute/60. + time_utc.second/3600. +
                time_utc.microsecond/3600.e6)

    # np.array needed for pandas > 0.20
    UnivDate = np.array(DayOfYear)
    UnivHr = np.array(DecHours)

    Yr = np.array(time_utc.year) - 1900
    YrBegin = 365 * Yr + np.floor((Yr - 1) / 4.) - 0.5

    Ezero = YrBegin + UnivDate
    T = Ezero / 36525.

    # Calculate Greenwich Mean Sidereal Time (GMST)
    GMST0 = 6 / 24. + 38 / 1440. + (
        45.836 + 8640184.542 * T + 0.0929 * T ** 2) / 86400.
    GMST0 = 360 * (GMST0 - np.floor(GMST0))
    GMSTi = np.mod(GMST0 + 360 * (1.0027379093 * UnivHr / 24.), 360)

    # Local apparent sidereal time
    LocAST = np.mod((360 + GMSTi - Longitude), 360)

    EpochDate = Ezero + UnivHr / 24.
    T1 = EpochDate / 36525.

    ObliquityR = np.radians(
        23.452294 - 0.0130125 * T1 - 1.64e-06 * T1 ** 2 + 5.03e-07 * T1 ** 3)
    MlPerigee = 281.22083 + 4.70684e-05 * EpochDate + 0.000453 * T1 ** 2 + (
        3e-06 * T1 ** 3)
    MeanAnom = np.mod((358.47583 + 0.985600267 * EpochDate - 0.00015 *
                       T1 ** 2 - 3e-06 * T1 ** 3), 360)
    Eccen = 0.01675104 - 4.18e-05 * T1 - 1.26e-07 * T1 ** 2
    EccenAnom = MeanAnom
    E = 0

    while np.max(abs(EccenAnom - E)) > 0.0001:
        E = EccenAnom
        EccenAnom = MeanAnom + np.degrees(Eccen)*np.sin(np.radians(E))

    TrueAnom = (
        2 * np.mod(np.degrees(np.arctan2(((1 + Eccen) / (1 - Eccen)) ** 0.5 *
                   np.tan(np.radians(EccenAnom) / 2.), 1)), 360))
    EcLon = np.mod(MlPerigee + TrueAnom, 360) - Abber
    EcLonR = np.radians(EcLon)
    DecR = np.arcsin(np.sin(ObliquityR)*np.sin(EcLonR))

    RtAscen = np.degrees(np.arctan2(np.cos(ObliquityR)*np.sin(EcLonR),
                                    np.cos(EcLonR)))

    HrAngle = LocAST - RtAscen
    HrAngleR = np.radians(HrAngle)
    HrAngle = HrAngle - (360 * ((abs(HrAngle) > 180)))

    SunAz = np.degrees(np.arctan2(-np.sin(HrAngleR),
                                  np.cos(LatR)*np.tan(DecR) -
                                  np.sin(LatR)*np.cos(HrAngleR)))
    SunAz[SunAz < 0] += 360

    SunEl = np.degrees(np.arcsin(
        np.cos(LatR) * np.cos(DecR) * np.cos(HrAngleR) +
        np.sin(LatR) * np.sin(DecR)))

    SolarTime = (180 + HrAngle) / 15.

    # Calculate refraction correction
    Elevation = SunEl
    TanEl = pd.Series(np.tan(np.radians(Elevation)), index=time_utc)
    Refract = pd.Series(0, index=time_utc)

    Refract[(Elevation > 5) & (Elevation <= 85)] = (
        58.1/TanEl - 0.07/(TanEl**3) + 8.6e-05/(TanEl**5))

    Refract[(Elevation > -0.575) & (Elevation <= 5)] = (
        Elevation *
        (-518.2 + Elevation*(103.4 + Elevation*(-12.79 + Elevation*0.711))) +
        1735)

    Refract[(Elevation > -1) & (Elevation <= -0.575)] = -20.774 / TanEl

    Refract *= (283/(273. + temperature)) * (pressure/101325.) / 3600.

    ApparentSunEl = SunEl + Refract

    # make output DataFrame
    DFOut = pd.DataFrame(index=time)
    DFOut['apparent_elevation'] = ApparentSunEl
    DFOut['elevation'] = SunEl
    DFOut['azimuth'] = SunAz
    DFOut['apparent_zenith'] = 90 - ApparentSunEl
    DFOut['zenith'] = 90 - SunEl
    DFOut['solar_time'] = SolarTime

    return DFOut


def calc_time(lower_bound, upper_bound, latitude, longitude, attribute, value,
              altitude=0, pressure=101325, temperature=12, xtol=1.0e-12):
   
    try:
        import scipy.optimize as so
    except ImportError:
        raise ImportError('The calc_time function requires scipy')

    obs, sun = _ephem_setup(latitude, longitude, altitude,
                            pressure, temperature)

    def compute_attr(thetime, target, attr):
        obs.date = thetime
        sun.compute(obs)
        return getattr(sun, attr) - target

    lb = datetime_to_djd(lower_bound)
    ub = datetime_to_djd(upper_bound)

    djd_root = so.brentq(compute_attr, lb, ub,
                         (value, attribute), xtol=xtol)

    return djd_to_datetime(djd_root)


def pyephem_earthsun_distance(time):

    import ephem

    sun = ephem.Sun()
    earthsun = []
    for thetime in time:
        sun.compute(ephem.Date(thetime))
        earthsun.append(sun.earth_distance)

    return pd.Series(earthsun, index=time)


def nrel_earthsun_distance(time, how='numpy', delta_t=67.0, numthreads=4):
   
    if not isinstance(time, pd.DatetimeIndex):
        try:
            time = pd.DatetimeIndex(time)
        except (TypeError, ValueError):
            time = pd.DatetimeIndex([time, ])

    unixtime = np.array(time.astype(np.int64)/10**9)

    spa = _spa_python_import(how)

    delta_t = delta_t or spa.calculate_deltat(time.year, time.month)

    dist = spa.earthsun_distance(unixtime, delta_t, numthreads)

    dist = pd.Series(dist, index=time)

    return dist


def _calculate_simple_day_angle(dayofyear):
    return (2. * np.pi / 365.) * (dayofyear - 1)


def equation_of_time_spencer71(dayofyear):
    day_angle = _calculate_simple_day_angle(dayofyear)
    # convert from radians to minutes per day = 24[h/day] * 60[min/h] / 2 / pi
    return (1440.0 / 2 / np.pi) * (0.0000075 +
        0.001868 * np.cos(day_angle) - 0.032077 * np.sin(day_angle) -
        0.014615 * np.cos(2.0 * day_angle) - 0.040849 * np.sin(2.0 * day_angle)
    )


def equation_of_time_pvcdrom(dayofyear):
    """
    Equation of time from PVCDROM.

    `PVCDROM`_ is a website by Solar Power Lab at Arizona State University (ASU)

    .. _PVCDROM: http://www.pveducation.org/pvcdrom/2-properties-sunlight/solar-time

    Parameters
    ----------
    dayofyear : numeric

    Returns
    -------
    equation_of_time : numeric
        Difference in time between solar time and mean solar time in minutes.

    References
    ----------
    [1] Soteris A. Kalogirou, "Solar Energy Engineering Processes and Systems,
    2nd Edition" Elselvier/Academic Press (2009).

    See Also
    --------
    equation_of_time_Spencer71
    """
    # day angle relative to Vernal Equinox, typically March 22 (day number 81)
    bday = _calculate_simple_day_angle(dayofyear) - (2.0 * np.pi / 365.0) * 80.0
    # same value but about 2x faster than Spencer (1971)
    return 9.87 * np.sin(2.0 * bday) - 7.53 * np.cos(bday) - 1.5 * np.sin(bday)


def declination_spencer71(dayofyear):
    day_angle = _calculate_simple_day_angle(dayofyear)
    return (0.006918 -
        0.399912 * np.cos(day_angle) + 0.070257 * np.sin(day_angle) -
        0.006758 * np.cos(2. * day_angle) + 0.000907 * np.sin(2. * day_angle) -
        0.002697 * np.cos(3. * day_angle) + 0.00148 * np.sin(3. * day_angle)
    )


def declination_cooper69(dayofyear):
    day_angle = _calculate_simple_day_angle(dayofyear)
    return np.deg2rad(23.45 * np.sin(day_angle + (2.0 * np.pi / 365.0) * 285.0))


def solar_azimuth_analytical(latitude, hour_angle, declination, zenith):


    numer = (np.cos(zenith) * np.sin(latitude) - np.sin(declination))
    denom = (np.sin(zenith) * np.cos(latitude))

    # cases that would generate new NaN values are safely ignored here
    # since they are dealt with further below
    with np.errstate(invalid='ignore', divide='ignore'):
        cos_azi = numer / denom

    # when zero division occurs, use the limit value of the analytical expression
    cos_azi = np.where(np.isclose(denom,    0.0, rtol=0.0, atol=1e-8),  1.0, cos_azi)

    # when too many round-ups in floating point math take cos_azi beyond 1.0, use 1.0
    cos_azi = np.where(np.isclose(cos_azi,  1.0, rtol=0.0, atol=1e-8),  1.0, cos_azi)
    cos_azi = np.where(np.isclose(cos_azi, -1.0, rtol=0.0, atol=1e-8), -1.0, cos_azi)

    # when NaN values occur in input, ignore and pass to output
    with np.errstate(invalid='ignore'):
        sign_ha = np.sign(hour_angle)

    return (sign_ha * np.arccos(cos_azi) + np.pi)


def solar_zenith_analytical(latitude, hour_angle, declination):
 
    return np.arccos(
        np.cos(declination) * np.cos(latitude) * np.cos(hour_angle) +
        np.sin(declination) * np.sin(latitude)
    )


def hour_angle(times, longitude, equation_of_time):
 
    hours = np.array([(t - t.tz.localize(
        dt.datetime(t.year, t.month, t.day)
    )).total_seconds() / 3600. for t in times])
    timezone = times.tz.utcoffset(times).total_seconds() / 3600.
    return 15. * (hours - 12. - timezone) + longitude + equation_of_time / 4.