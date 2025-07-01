def sim_ws_wd_series(self, t_array, ws_array, wd_array,
                     rotor_rpm_init=10,
                     init_pitch=0.0,
                     init_yaw=None,
                     make_plots=True):
    '''
    Simulate simplified turbine model using a complied controller (.dll or similar).
        - currently a 1DOF rotor model

    Parameters:
    -----------
        t_array: float
             Array of time steps, (s)
        ws_array: float
             Array of wind speeds, (s)
        wd_array: float
             Array of wind directions, (deg)
        rotor_rpm_init: float, optional
             Initial rotor speed, (rpm)
        init_pitch: float, optional
             Initial blade pitch angle, (deg)
        init_yaw: float, optional
             Initial yaw angle, if None then start with no misalignment,
             i.e., the yaw angle is set to the initial wind direction (deg)
        make_plots: bool, optional
             True: generate plots, False: don't. 
    '''

    # Store turbine data for convenience
    dt = t_array[1] - t_array[0]
    R = self.turbine.rotor_radius
    GBRatio = self.turbine.Ng

    # Declare output arrays
    bld_pitch = np.ones_like(t_array) * init_pitch
    rot_speed = np.ones_like(t_array) * rotor_rpm_init * \
        rpm2RadSec  # represent rot speed in rad / s
    gen_speed = np.ones_like(t_array) * rotor_rpm_init * GBRatio * \
        rpm2RadSec  # represent gen speed in rad/s
    aero_torque = np.ones_like(t_array) * 1000.0
    gen_torque = np.ones_like(t_array)  # * trq_cont(turbine_dict, gen_speed[0])
    gen_power = np.ones_like(t_array) * 0.0
    nac_yawerr = np.ones_like(t_array) * 0.0  #?
    if init_yaw is None:
        init_yaw = wd_array[0]
    else:
        nac_yawerr[0] = init_yaw - wd_array[0]
    nac_yaw = np.ones_like(t_array) * init_yaw
    nac_yawrate = np.ones_like(t_array) * 0.0   #?
    
    
    #function for calculate yaw_error
    def yaw_error(wind_dir, nacelle_dir):
        error = wind_dir - nacelle_dir
        error = ((error + 180) % 360) - 180
        return error
        

    # Loop through time
    for i, t in enumerate(t_array):
        if i == 0:
            continue  # Skip the first run
        ws = ws_array[i]
        wd = wd_array[i]
        nac_yawerr[i] = (wd - nac_yaw[i-1])*deg2rad
        
        print(nac_yawerr[i])

        # Load current Cq data
        tsr = rot_speed[i-1] * self.turbine.rotor_radius / ws
        cq = self.turbine.Cq.interp_surface([bld_pitch[i-1]], tsr)

        # Update the turbine state
        #       -- 1DOF model: rotor speed and generator speed (scaled by Ng)
        aero_torque[i] = 0.5 * self.turbine.rho * (np.pi * R**2) * cq * R * ws**2 * np.cos(nac_yawerr[i])**2  #Incorporate Yaw Misalignment in Aerodynamics: Add a cosine penalty to aero_torque (* np.cos(nac_yawerr[i])**2)
        rot_speed[i] = rot_speed[i-1] + (dt/self.turbine.J)*(aero_torque[i]
                                                             * self.turbine.GenEff/100 - self.turbine.Ng * gen_torque[i-1])
        gen_speed[i] = rot_speed[i] * self.turbine.Ng

        # populate turbine state dictionary
        turbine_state = {}
        # populate turbine state dictionary
        turbine_state = {}
        if i < len(t_array)-1:
            turbine_state['iStatus'] = 1
        else:
            turbine_state['iStatus'] = -1
        turbine_state['t'] = t
        turbine_state['dt'] = dt
        turbine_state['ws'] = ws
        turbine_state['bld_pitch'] = bld_pitch[i-1]
        turbine_state['gen_torque'] = gen_torque[i-1]
        turbine_state['gen_speed'] = gen_speed[i]
        turbine_state['gen_eff'] = self.turbine.GenEff/100
        turbine_state['rot_speed'] = rot_speed[i]
        turbine_state['Yaw_fromNorth'] = nac_yaw[i]
        turbine_state['Y_MeasErr'] = nac_yawerr[i-1]
        
        # Call the controller

        gen_torque[i], bld_pitch[i], nac_yawrate[i] = self.controller_int.call_controller(turbine_state)

        # Calculate the power
        gen_power[i] = gen_speed[i] * gen_torque[i]
        gen_power[i] = gen_speed[i] * gen_torque[i] * self.turbine.GenEff / 100

        # Update the nacelle position
        nac_yaw[i] = nac_yaw[i-1] + nac_yawrate[i]*rad2deg*dt
        
        
    self.controller_int.kill_discon()

    # Save these values
    self.bld_pitch = bld_pitch
    self.rot_speed = rot_speed
    self.gen_speed = gen_speed
    self.aero_torque = aero_torque
    self.gen_torque = gen_torque
    self.gen_power = gen_power
    self.t_array = t_array
    self.ws_array = ws_array
    self.wd_array = wd_array
    self.nac_yaw = nac_yaw
    self.nac_yawrate = nac_yawrate
    
    
    if make_plots:
        fig, axarr = plt.subplots(nrows=6, sharex=True, figsize=(8, 14))

        ax = axarr[0]
        ax.plot(self.t_array, self.ws_array)
        ax.set_ylabel('Wind Speed (m/s)')

        ax = axarr[1]
        ax.plot(self.t_array, self.wd_array, label='wind direction')
        ax.plot(self.t_array, self.nac_yaw, label='yaw position')
        ax.set_ylabel('Wind Direction (deg)')
        ax.legend(loc='best')

        ax = axarr[2]
        #ax.plot(self.t_array, self.wd_array-self.nac_yaw)#*rad2deg
        my_yaw_error = yaw_error(wind_dir=self.wd_array, nacelle_dir=self.nac_yaw)
        ax.plot(self.t_array, my_yaw_error)
        ax.set_ylabel('Nacelle yaw error (deg)')
        
        ax = axarr[3]
        ax.plot(self.t_array, self.rot_speed)
        ax.set_ylabel('Rot Speed (rad/s)')

        ax = axarr[4]
        ax.plot(self.t_array, self.gen_torque)
        ax.set_ylabel('Gen Torque (N)')

        ax = axarr[5]
        ax.plot(self.t_array, self.bld_pitch*rad2deg)
        ax.set_ylabel('Bld Pitch (deg)')
                   
        ax.set_xlabel('Time (s)')
        for ax in axarr:
            ax.grid()