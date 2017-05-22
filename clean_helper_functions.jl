## Functions for linear interpolation ##

# Lowest level function - interpolates between two points
function interp_pts(x1, y1, x2, y2, x0)
  if(x2-x1 == 0)
    return (y1+y2)/2
  end
  slope = (y2-y1)/(x2-x1)
  return y1 + slope*(x0-x1)
end

# Interpolates between two vectors
# Assume that the time vectors are sorted.
function interp1(new_time, old_data, old_time)
  new_data = zeros(size(new_time,1),size(old_data,2))
  old_ind = 1
  new_ind = 1

  while(new_ind <= size(new_data,1))
    # Get started 
    i = findfirst( old_time.> new_time[new_ind])
    if(i == 0)
      println("Warning: new_time not contained in old time")
      if(old_ind == 1)
        return []
      else
        new_data[new_ind,:] = new_data[new_ind-1,:]
      end 
    elseif(i==1)
      println("Warning: No history")
      new_data[new_ind,:] = old_data[i,:]
    else
      new_data[new_ind,:] = interp_pts(old_time[i-1], old_data[i-1,:], old_time[i], old_data[i,:], new_time[new_ind])
    end
    old_ind +=1 
    new_ind +=1
  end
  return new_data
end

## Fucntions for loading data ##
function load_features()
    data = readcsv("features.csv")
    t = data[:,1]
    t -= t[1]
    return t, data[:,2]
end

function load_data()
    data = readcsv("_slash_tf.csv")
    # strip out headers
    headers = data[1,:] # Note there's an extra "-" header that shifts things
    data = data[2:end,:]
    # strip out useful information:
    t = data[:,1]*(1.0e-9);       # rosbag timestamps - units S
    t -= t[1]
    frame_id = data[:,9] # This is actually "child_frame_id", but they're still unique
    # Warning: some of these values have empty strings, so a typecast may not work.
    tra = data[:,12:14] # Translation: XYZ
    rot = data[:,15:18] # Rotation: XYZW

    # sort the data by frame:
    vicon_inds = find(frame_id.=="vicon/quad_1/quad_1")
    tango_inds = find(frame_id.=="device")
    sos_inds    = find(frame_id.=="start_of_service")

    # Code from Brian - thanks!
    # First 1000 measurements or so are of the same position
    start_inds = 1:100
    # Compute transform using least squares
    # Write up here: https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
    # A = p, B = q in their notation.
    A = tra[tango_inds[start_inds],:]# rot[tango_inds[start_inds]]]
    B = tra[vicon_inds[start_inds],:]# rot[vicon_inds[start_inds]]]

    centroid_A = [mean(A[:,1]); mean(A[:,2]); mean(A[:,3])]
    centroid_B = [mean(B[:,1]); mean(B[:,2]); mean(B[:,3])]
    N = size(A,1)
    H = (A - repmat(centroid_A',N,1))' * (B - repmat(centroid_B', N, 1))
    H = convert(Array{Float64,2},H)
    U, S, V = svd(float(H))
    R = V*U'
    if det(R) < 0
        V[:,3] *= -1
        R = V*U'
    end

    offset = -R*centroid_A + centroid_B

    # transformed coordinates:
    tra_t = broadcast(+,(R*(tra[tango_inds,:]')),offset)'


    # This rotation is getting screwed up.
    # Want: swap y/z
    # change sign on x
    # which is a rotation by (pi,pi/2,0)

    # This is all manual adjustment to make things look better - 
    # should be removed.
    tra_t[:,1] = 2*tra_t[1,1] - (tra_t[:,1])
    tra_t[:,3] -= tra_t[1,3]

    # Interpolate so on the same scale.
    v_interp = interp1(t[tango_inds], tra[vicon_inds,:], t[vicon_inds]);

    return t[tango_inds], tra_t, v_interp
end

function estimate_error(t, tra_tango, tra_vicon)
    # Estimate error rate using an exponential filter:
    error = tra_vicon - tra_tango;

    error_rate = zeros(size(error))
    t_elapsed = 0;
    rate = 0.999
    
    dx_last = 0;
    for (iter,) in enumerate(t)
       if(iter==1)
         error_rate[iter] =  0
         v_last = 0
       else
         dx_inst = (error[iter,:]-error[iter-1,:])
         dx_last = rate*dx_last + (1-rate)*dx_inst

         dt = t[iter]-t[iter-1] 
         t_elapsed = rate*t_elapsed + (1-rate)*dt # average number of measurements
         error_rate[iter,:] = dx_last/t_elapsed #rate*error_rate[iter-1] + (1-rate)*v_inst
       end
    end
    return error, error_rate
end


