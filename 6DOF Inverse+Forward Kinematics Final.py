import math
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

def xyzInverseSolve(pX, pY, pZ, facing, solution): #converts x y z pos to axis positions

    solA = math.atan2(pX, pY) #axis 1 angle, "points" at x y position
    if facing != True: solA -= math.copysign(180, solA) #if position "behind" axis 2 turn 180

    lX = math.sqrt(pX ** 2 + pY ** 2) #top down distance between center of axis 1 rotation to x y point
    if facing != True: lX *= -1 #if position solution "behind" axis 2
    lY = pZ #arm plane y position, equal to z position
    r = math.sqrt(lX ** 2 + lY ** 2) #arm plane distance betwen axis 2 and point
    if r > (rA + rB):
        print("point is out of arms reach")
        return #is point out of reach (distance larger then combined arm length)

    n = (r ** 2 + rA ** 2 - rB ** 2) / 2 #term used in quadratic multiple times
    xTerm = (n * lX) / (r ** 2) #-b term of quadratic formula, over 2a for x coordinate
    yTerm = (n * lY) / (r ** 2) #-b term of quadratic formula, over 2a for y cooridnate
    xRoot = math.sqrt((n ** 2 * lX ** 2) - (r ** 2 * (n ** 2 - (lY ** 2 * rA ** 2)))) / (r ** 2) #root b^2-4ac term of quadratic, over 2a for x coordinate
    yRoot = math.sqrt((n ** 2 * lY ** 2) - (r ** 2 * (n ** 2 - (lX ** 2 * rA ** 2)))) / (r ** 2) #root b^2-4ac term of quadratic, over 2a for y coordinate
        
    if solution == True:
        x1 = xTerm - math.copysign(xRoot, lY) #swaps order of solutions when arm plane y < 0
        y1 = yTerm + math.copysign(yRoot, lX) #swaps order of solutions when arm plane x < 0
    else:
        x1 = xTerm + math.copysign(xRoot, lY) #swaps order of solutions when arm plane y < 0
        y1 = yTerm - math.copysign(yRoot, lX) #swaps order of solutions when arm plane x < 0

    x2 = lX - x1
    y2 = lY - y1
    solB = math.atan2(y1, x1)
    solC = math.atan2(y2, x2)
    return solA, solB, solC

def magnitude(vector): #calculates the magnitude of a vector
    return math.sqrt(sum(pow(element, 2) for element in vector))

def cosVectorAngle(A, B): #cosine vector angle formula, measures 0-180
    return math.acos(np.dot(A, B) / ((magnitude(A) * magnitude(B))))

def sinVectorAngle(A, B): #sin vector angle formula, measures 0-180
    return math.asin(magnitude(np.cross(A, B)) / ((magnitude(A) * magnitude(B))))

def unitVector(vector): #returns vector of unit (1) length/magnitude with same direction as input
    return np.divide(vector, magnitude(vector))

def normalVectorAngle(normal, A, B): #calculates vector angle about a given axis (or plane) such that the angle is always measured in the same direction from -180 to 180
    determinant = np.dot(unitVector(normal), np.cross(A, B))
    dot = np.dot(A, B)
    return math.atan2(determinant, dot)

def normalizeAngle(angle): #takes in an angle of arbirtary magnitude and puts it between -pi and +pi (-180 to 180)
    return (angle + math.pi) % math.tau - math.pi

def compareOrient(last, target): #returns the biggest absolute difference between two orientation vectors even between the 180deg point
    offset = np.multiply(math.tau, [1, 0, 1]) #second element is 0 because ax5 cannot rotate through 180deg
    last = [normalizeAngle(last[0]), normalizeAngle(last[1]), normalizeAngle(last[2])] #input angles need to be between -180 and 180 for this function to work
    pos = target #positions without offset
    positiveOffsetPos = np.add(target, offset) #positions with +360 offset
    negativeOffsetPos = np.subtract(target, offset) #positions with -360 offset

    normalDiff = np.subtract(pos, last) #compares the vectors standardly
    positiveOffsetDiff = np.subtract(positiveOffsetPos, last) #adds a target +-360 deg in every axis, as there may be an easier way going through 180deg
    negativeOffsetDiff = np.subtract(negativeOffsetPos, last) #for example if we are at -179 and we want to go to 179, we dont want to go 358 degrees CCW but 2deg CW. If we subtract 360 from the target, we see that -181 is where we actually want to go

    diff = [positiveOffsetDiff, negativeOffsetDiff, normalDiff] #arranges the differences into a 3x3 matrix where the columns are axes
    posList = [positiveOffsetPos, negativeOffsetPos, pos] #creates array of offset positions and normal position
    diff = np.transpose(diff) #swaps columns and rows so each axis is on the same row for easy comparision
    posList = np.transpose(posList) #does the same thing with this one so they match up

    # absolute = [min(abs(element)) for element in diff] #takes the minimum absoulute difference for each axis as the shortest path between position
    absolute = np.min(np.abs(diff), axis=1) #returns the lowest absolute value of each row as the shortest possible path for that axis
    closest = posList[np.arange(diff.shape[0]), np.argmin(np.abs(diff), axis=1)] #uses the indicies of the minimum abs values of each row to find the closest position
    signs = np.sign(diff[np.arange(diff.shape[0]), np.argmin(np.abs(diff), axis=1)]) #returns the sign (direction) that each axis should take to the closest position
    return max(absolute), closest #returns the biggest difference for any axis between vector positions while taking the shortest direction

def inverseSolve(tx, ty, tz, azimuth, elevation, roll, facing, solution):
    target = [tx, ty, tz] #location of the end of the end effector

    dir = np.multiply(rC, [math.sin(math.pi / 2 - elevation) * math.cos(azimuth), math.sin(math.pi / 2 - elevation) * math.sin(azimuth), math.cos(math.pi / 2 - elevation)]) #the direction that the end effector is pointing
    wrist = target - dir #vector to 3DOF solution at end of forearm
    ax1, ax2, ax3 = xyzInverseSolve(wrist[0], wrist[1], wrist[2], facing, solution) #calculates simple solution to 3DOF kinematics
    elbow = np.multiply(rA, [math.sin(math.pi / 2 - ax2)  * math.sin(ax1), math.sin(math.pi / 2 - ax2) * math.cos(ax1), math.cos(math.pi / 2 - ax2)]) 
    forearm = wrist - elbow #forearm vector direction by subtraction of elbow from wrist

    projected_elbow_roll = dir - np.multiply(forearm, np.dot(dir, forearm) / (magnitude(forearm) * magnitude(forearm))) #projects the end effector direction onto the plane normal to the forearm and on the wrist point so it can easily be used to calculate elbow roll
    inverse_projected_elbow_roll = -1.0 * projected_elbow_roll #when wrist bent backwards, angle measured from the opposite for consistency
    arm_plane_normal = np.cross((0, 0, 1), wrist) #determines the normal vector of the planar section of the arm (upper arm and forearm)

    ax4_forward = normalVectorAngle(forearm, arm_plane_normal, projected_elbow_roll) #calculates the angle between the projected end effector and the arm plane normal vector, about the forearm
    ax4_reverse = normalVectorAngle(forearm, arm_plane_normal, inverse_projected_elbow_roll) #if the wrist is bent backwards, we must measure from the inverse of the projected or it will be 180deg off

    ax5_forward_reference = np.cross(forearm, projected_elbow_roll) #angle measurement reference for ax5
    ax5_reverse_refernce = np.cross(forearm, inverse_projected_elbow_roll) #angle measurement reference for bent backwards ax5
    ax5_forward = normalVectorAngle(ax5_forward_reference, forearm, dir) #calculates the angle between the forearm and end effector vector, about the perpendicular fixed vector
    ax5_reverse = normalVectorAngle(ax5_reverse_refernce, forearm, dir)

    wrist_forward_reference = np.cross(forearm, dir) #calculates the plane/vector on the forearm and end effector which the roll that the motor does is measured from
    wrist_reverse_reference = -1.0 * wrist_forward_reference #in case that the wrist is bending backwards, it must be inverted to stay the same direction due to the cross product flipping around
    roll_horizontial_vector = unitVector(np.cross((0, 0, 1), dir)) #horizontal reference for calculating the absolute roll direction
    roll_vertical_vector = unitVector(np.cross(dir, roll_horizontial_vector)) #vertical reference for calculating absoulute roll 
    
    roll_direction_horizontal = unitVector(math.cos(roll) * roll_horizontial_vector + math.sin(roll) * roll_vertical_vector) #the absolute roll direction, used for math but also gripper oriented control left right
    roll_direction_vertical = unitVector(np.cross(dir, roll_direction_horizontal)) #gripper oriented control direction up down

    vector_length_ratio = math.pow(10, 3) #how long the offset angle measurement vector should be relative to the unit direction vector. Should theoretically be infinitely small
    orient_vector_horizontal = unitVector(dir) + np.divide(roll_direction_horizontal, vector_length_ratio) #offsets the direction vector by the horizontial direction to calculate a new orientation relative to the gripper plane
    orient_vector_vertical = unitVector(dir) + np.divide(roll_direction_vertical, vector_length_ratio) #offsets the direction vector by the vertical direction to calculate a new orientaition relative to the gripper plane
    orientation = [normalizeAngle(azimuth), normalizeAngle(elevation), normalizeAngle(roll)] #resets orientation to within -180 to 180 to prevent values massively growing at certain regions
    orient_control_horizontial = np.divide(np.subtract(getAbsoluteOrientation(orient_vector_horizontal, roll_direction_horizontal), orientation), math.asin(1 / vector_length_ratio)) #calculates absoulute orientation of new vector, subtracts from original orientation to get an angle difference, then divide by the theortical angle between the vectors (tan(theta) = opposite / 
    orient_control_vertical = np.divide(np.subtract(getAbsoluteOrientation(orient_vector_vertical, roll_direction_horizontal), orientation), math.asin(1 / vector_length_ratio))

    roll_direction_depth = unitVector(dir) #gripper oriented control direction in out
    ax6_forward = normalVectorAngle(dir, roll_direction_horizontal, wrist_forward_reference) #calculates roll relative to the ax5 tilt plane
    ax6_reverse = normalVectorAngle(dir, roll_direction_horizontal, wrist_reverse_reference) #uses opposite axis due to reversed projected elbow roll
    # global vectors
    # graphVector([0, 0, 0], target)
    # graphVector([0, 0, 0], elbow)
    # graphVector(elbow, forearm)
    # graphVector(wrist, dir)
    # graphVector(target, roll_direction_horozontial * rC / 2)
  
    return [ax1, ax2, ax3], [ax4_forward, ax5_forward, ax6_forward], [ax4_reverse, ax5_reverse, ax6_reverse], [roll_direction_horizontal, roll_direction_vertical, roll_direction_depth], [orient_control_horizontial, orient_control_vertical, np.array((0, 0, 1.0))]

def forwardSolve(ax1, ax2, ax3, ax4, ax5, ax6):
    elbow = np.multiply(rA, [math.sin(math.pi / 2 - ax2) * math.sin(ax1), math.sin(math.pi / 2 - ax2) * math.cos(ax1), math.cos(math.pi / 2 - ax2)]) #calculates elbow using spherical coords
    forearm = np.multiply(rB, [math.sin(math.pi / 2 - ax3) * math.sin(ax1), math.sin(math.pi / 2 - ax3) * math.cos(ax1), math.cos(math.pi / 2 - ax3)]) #calculates forearm using spherical coords
    wrist = elbow + forearm #combine elbow and forearm to get wrist position
    elbow_roll_horozontial_reference = unitVector(np.cross((0, 0, 1), wrist)) #x vector relative to the end of the forearm, perpendicular to the plane of the arm
    elbow_roll_vertical_reference = unitVector(np.cross(forearm, elbow_roll_horozontial_reference)) #y vector relative to end of the forearm, in the plane of the arm
    dir = np.multiply(rC, elbow_roll_vertical_reference * math.sin(ax5) * math.sin(ax4) + elbow_roll_horozontial_reference * math.sin(ax5) * math.cos(ax4) + unitVector(forearm) * math.cos(ax5)) #calculates the end effector direction vector using the above 2 vectors and forearm direction
    roll_plane_normal = unitVector(np.cross(forearm, dir)) #x direction for roll relative to the arm
    projected_elbow_roll = dir - np.multiply(forearm, np.dot(dir, forearm) / (magnitude(forearm) * magnitude(forearm))) #same as inverse solver
    if ax5 < 0: #if wrist is bent backwards, inverts the measurement references
        roll_plane_normal *= -1.0
        projected_elbow_roll *= -1.0
    roll_bend_normal = unitVector(np.cross(roll_plane_normal, dir)) #y direction for roll relative to the arm
    roll_vector = unitVector(roll_plane_normal * math.cos(ax6) + roll_bend_normal * math.sin(ax6)) #calculates the perpendicular vector that indicates roll using spherical coords and the previous vectors
    roll_normal_vector = unitVector(np.cross(dir, roll_vector))
    roll_horizontial_reference = np.cross((0, 0, 1), dir) #universal measurement reference for absolute roll
    target = wrist + dir #final XYZ position which should match with the inverse input
    # twist = normalVectorAngle(forearm, elbow_roll_horozontial_reference, projected_elbow_roll)
    # bend = normalVectorAngle(roll_plane_normal, forearm, dir)
    # rotation = normalVectorAngle(dir, roll_vector, roll_plane_normal)
    # print(math.degrees(twist), math.degrees(bend), math.degrees(rotation))
    orientation = getAbsoluteOrientation(dir, roll_vector)
    global vectors
    graphVector([0, 0, 0], elbow)
    graphVector(elbow, forearm)
    graphVector(wrist, dir)
    # graphVector(wrist, unitVector(elbow_roll_horozontial_reference) * rC / 2)
    # graphVector(wrist, unitVector(projected_elbow_roll) * rC / 2)
    graphVector(target, roll_vector * rC / 2)
    graphVector(target, roll_normal_vector * rC / 4)
    return target, orientation

def getAbsoluteOrientation(direction, roll_direction):
    horizontial_reference = np.cross((0, 0, 1), direction) #creates horizontial reference 
    projected_direction = direction - np.multiply((0, 0, 1), np.dot(direction, (0, 0, 1))) #projects direction onto xy plane
    azimuth = normalVectorAngle((0, 0, 1), (1, 0, 0), projected_direction) #measures angle between horizontial reference and projected direction about z axis
    elevation = normalVectorAngle(horizontial_reference, direction, projected_direction) #measures angle between direction and projected reference about horizontial reference
    roll = normalVectorAngle(direction, horizontial_reference, roll_direction) #measures roll about direction from horizontial reference
    return [azimuth, elevation, roll]


def graphVector(origin, end): #adds vector to 3d plot (origin is absolute, end is relative to origin)
    global vectors
    vectors = np.vstack((vectors, np.concatenate([list(origin), list(end)])))

def solve():
    global rollOrientControl, endEffectorOriented, vectors, pos, orientation, posResult, orientResult, motorDir, current

    vectors = np.empty((0, 6))
    segments, forwardDirection, reverseDirection, endEffectorOriented, rollOrientControl = inverseSolve(pos[0], pos[1], pos[2], math.radians(orientation[0]), math.radians(orientation[1]), math.radians(orientation[2]), True, True)
    axis1, axis2, axis3 = segments[0], segments[1], segments[2] #xyz is always the same besides the forward/inverse solution
    forwardMax, forwardDir = compareOrient(current, forwardDirection)
    reverseMax, reverseDir = compareOrient(current, reverseDirection)
    if forwardMax < reverseMax : #compares both configurations to determine which the robot is currently closest to
        # axis4, axis5, axis6 = forwardDirection[0], forwardDirection[1], forwardDirection[2]
        axis4, axis5, axis6 = forwardDir
        print("forward")
    else:
        # axis4, axis5, axis6 = reverseDirection[0], reverseDirection[1], reverseDirection[2]
        axis4, axis5, axis6 = reverseDir
        print("reverse")

    current = axis4, axis5, axis6
    print(math.degrees(axis1), ",", math.degrees(axis2), ",", math.degrees(axis3), ",", math.degrees(axis4), ",", math.degrees(axis5), ",", math.degrees(axis6))
    # print(forwardDirection)
    # print(reverseDirection)

    posResult, orientResult = forwardSolve(axis1, axis2, axis3, axis4, axis5, axis6)
    print(posResult[0], posResult[1], posResult[2], math.degrees(orientResult[0]), math.degrees(orientResult[1]), math.degrees(orientResult[2]))

    # rollOrientControl = [np.array((math.cos(orientation[2]), math.sin(orientation[2]), 0)), np.array((math.sin(orientation[2]), math.cos(orientation[2]), 0)), np.array((0, 0, 1.0))]

def update(frame):
    global vectors, X, Y, Z, U, V, W
    solve()
    ax.cla()
    X, Y, Z, U, V, W = zip(*vectors)
    ax.quiver(X, Y, Z, U, V, W, arrow_length_ratio=0.0)
    # Set the axis limits
    ax.set_xlim([-size, size])
    ax.set_ylim([-size, size])
    ax.set_zlim([-size, size])

    # Redraw the plot
    fig.canvas.draw()
    plt.show()


def update_pos(delta):
    pos[0] += delta[0]
    pos[1] += delta[1]
    pos[2] += delta[2]

def update_orientation(delta):
    orientation[0] += delta[0]
    orientation[1] += delta[1]
    orientation[2] += delta[2]

def disable_key_shortcuts():
    plt.rcParams['keymap.save'].remove('s')
    plt.rcParams['keymap.fullscreen'].remove('f')
    plt.rcParams['keymap.back'].remove('c')
    plt.rcParams['keymap.forward'].remove('v')
    plt.rcParams['keymap.grid'].remove('g')
    plt.rcParams['keymap.home'].remove('h')
    plt.rcParams['keymap.home'].remove('r')
    plt.rcParams['keymap.pan'].remove('p')
    plt.rcParams['keymap.quit'].remove('q')
    plt.rcParams['keymap.xscale'].remove('k')
    plt.rcParams['keymap.yscale'].remove('l')
    plt.rcParams['keymap.zoom'].remove('o')

def analogInputParse(pos_rates, rot_rates, relative, move_speed, tilt_speed):
    global endEffectorOriented, rollOrientControl
    if relative:
        delta_pos = np.add(np.multiply(pos_rates[0], endEffectorOriented[0]), np.multiply(pos_rates[1], endEffectorOriented[1]), np.multiply(pos_rates[2], endEffectorOriented[2]))
        delta_rot = np.add(np.multiply(rot_rates[0], rollOrientControl[0]), np.multiply(rot_rates[1], rollOrientControl[1]), np.multiply(rot_rates[2], rollOrientControl[2]))
    else:
        delta_pos = pos_rates
        delta_rot = rot_rates
    update_pos(np.multiply(move_speed, delta_pos))
    update_orientation(np.multiply(tilt_speed, delta_rot))


def on_key_press(event):
    global endEffectorOriented, rollOrientControl
    solve()
    key = event.key
    joystickX, joystickY, joystickZ, joystickAzimuth, joystickElevation, joystickRoll = 0, 0, 0, 0, 0, 0
    analogInputParse((joystickX, joystickY, joystickZ), (joystickAzimuth, joystickElevation, joystickRoll), True, move_speed, tilt_speed)
    switch = {
#movement controls
#side to side
        'a': lambda: update_pos(move_speed * endEffectorOriented[0]),
        'd': lambda: update_pos(move_speed * -1.0 * endEffectorOriented[0]),
#up down
        'w': lambda: update_pos(move_speed * endEffectorOriented[1]),
        's': lambda: update_pos(move_speed * -1.0 * endEffectorOriented[1]),
#in out
        'q': lambda: update_pos(move_speed * endEffectorOriented[2]),
        'e': lambda: update_pos(move_speed * -1.0 * endEffectorOriented[2]),

#orientation controls
#azimuth
        'f': lambda: update_orientation(tilt_speed * rollOrientControl[0]),
        'h': lambda: update_orientation(tilt_speed * -1.0 * rollOrientControl[0]),
#elevation
        'g': lambda: update_orientation(tilt_speed * rollOrientControl[1]),
        't': lambda: update_orientation(tilt_speed * -1.0 * rollOrientControl[1]),
#roll
        'r': lambda: update_orientation(tilt_speed * rollOrientControl[2]),
        'y': lambda: update_orientation(tilt_speed * -1.0 * rollOrientControl[2]),

#circle
        'c': lambda: traceCircle(circleCenter, circleDirection, circleRadius, theta_dot)
    }
    if key in switch:
        switch[key]()
        update(None)

def traceCircle(center, direction, radius, rate):
    global theta, pos
    theta += rate
    circleHorizontial = radius * unitVector(np.cross((0, 0, 1), direction))
    circleVertical = radius * unitVector(np.cross(circleHorizontial, direction))
    pos = math.sin(theta) * circleVertical + math.cos(theta) * circleHorizontial + center

global rollOrientControl, endEffectorOriented, vectors, pos, orientation, posResult, orientResult, motorDir, theta
vectors = np.empty((0, 6))    

rA = 0.4 #length of first segment
rB = 0.63 #length of second segment
rC = 0.127 #length of end effector segment
pos = [0.5, 0.0, 0.401]

orientation = [0, 0, 0]
current = [math.radians(0), math.radians(0), math.radians(0)] #last position of the end effector as motor angles (not absolute position)

segments = [] #ax1, ax2, ax3 angles
forwardDirection = [] #end effector (ax4, ax5, ax6) angles for when wrist is bent forward (ax5>0)
reverseDirection = [] #end effector angles for when wrist is bent backwards
endEffectorOriented = [] #orientation vectors for gripper-oriented control
rollOrientControl = []
motorDir = [] #direction that the end effector motors should take to get to the new configuration fastest
move_speed = 0.01
tilt_speed = 3.0

posResult = [] #xyz result from forward solver
orientResult = [] #orientation result from forward solver

theta = 0
theta_dot = 0.03
circleRadius = 0.25
circleCenter = [0.5, 0.25, 0]
circleDirection = [1, 0, 0]


disable_key_shortcuts()
size = (rA + rB + rC) / 2
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

fig.canvas.mpl_connect('key_press_event', on_key_press)
update(None)
plt.show()