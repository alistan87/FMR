@@ -0,0 +1,560 @@
import numpy as np
import matplotlib.pyplot as plt
import pickle
__author__ = 'AliSucipto'

"""
Helper functions to the class FMR
"""


def llg(m, H, alpha=0.006, gyrat=2.21e5):
    gamma = gyrat/(1 + alpha**2)
    torque = -gamma*(np.cross(m, H) + alpha*np.cross(m, np.cross(m, H)))
    return torque


class FMR:
    """
    Implementation of the ferromagnetic resonance class
    """

    def __init__(self, m01=(0, 1, 0), m02=(0, 1, 0)):
        self.m1 = m01
        self.m2 = m02
        self.t = 0
        self.phase = 0e-12
        self.dt = 1e-13
        self.demagTensor = np.array([[0, 0, 0],
                                     [0, 0, 0],
                                     [0, 0, 1]])
        self.alphaSC1 = 0.005
        self.alphaSC2 = 0.005

    def setRF(self, amplitude, frequency):
        self.rfAmplitude = amplitude
        self.rfFrequency = frequency

    def setSC(self, alpha1, alpha2):
        self.alphaSC1 = alpha1
        self.alphaSC2 = alpha2

    def getFieldRF(self):
        self.fieldRF = 1000*self.rfAmplitude/(4*np.pi)*np.asarray((np.cos(2*np.pi*self.rfFrequency*self.t),
                                        0, 0))
        return self.fieldRF

    def getDemag(self):
        """
        Thin film demagnetization tensor with normal direction along z-axis
        """
        ms1 = 8.0e5  # Py magnetization in A/m
        ms2 = 14.0e5 # Co magnetization in A/m
        self.fieldDemag1 = -ms1*np.dot(self.demagTensor, self.m1)
        # Include some number to test for the effect of surface anisotropy on Co
        self.fieldDemag2 = -(ms2 + 12.0e5)*np.dot(self.demagTensor, self.m2)
        return self.fieldDemag1, self.fieldDemag2

    def getanisfield(self):
        """
        Anisotropic field due to the surface anisotropy of the thin film
        """
        self.field

    def setFieldExt(self, field):
        if field == 0:
            self.fieldExt = (1000/(4*np.pi))*np.asarray(1e-3)
        else:
            self.fieldExt = (1000/(4*np.pi))*np.asarray(field)
        return self.fieldExt

    def setPhase(self, phase=0e-12):
        self.phase = phase

    def step(self, n=1):
        for i in range(n):
            self.H = self.getFieldRF() + self.fieldExt + self.getDemag()
            self.llg(self.m, self.H)
            self.m = self.m + self.torque*self.dt
            self.t += self.dt
            self.norm()
        return self.m

    def rk4(self):
        """
        Advance the magnetization using 4th order Runge-Kutta method
        """
        A = np.array([[0., 0., 0., 0.],
                      [1., 0., 0., 0.],
                      [0., 1., 0., 0.],
                      [0., 0., 1., 0.]])
        b = np.array([1./6, 1./3, 1./3, 1./6])
        c = np.array([0., 1./2, 1./2, 1.])

        k1 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        k2 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])

        # Compute the total effective field for each layer
        demag1, demag2 = self.getDemag()
        self.H1 = self.getFieldRF() + self.fieldExt + demag1
        self.H2 = self.getFieldRF() + self.fieldExt + demag2

        # Calculate the torque for the current field configuration using Runge-Kutta method
        for i in range(4):
            torque1 = np.array([0, 0, 0])
            torque2 = np.array([0, 0, 0])
            for j in range (4):
                torque1 = torque1 + A[i, j]*k1[j]
                torque2 = torque2 + A[i, j]*k2[j]
            k1[i] = llg(self.m1 + c[i]*self.dt*torque1, self.H1) # Calculate torque for Py
            k2[i] = llg(self.m2 + c[i]*self.dt*torque2, self.H2) # For Co

        dmdt1 = b[0]*k1[0] + b[1]*k1[1] + b[2]*k1[2] + b[3]*k1[3]
        dmdt2 = b[0]*k2[0] + b[1]*k2[1] + b[2]*k2[2] + b[3]*k2[3]

        self.dmdt1 = dmdt1 + self.alphaSC1*(np.cross(self.m1, dmdt1) - np.cross(self.m2, dmdt2))
        self.dmdt2 = dmdt2 + self.alphaSC2*(np.cross(self.m2, dmdt2) - np.cross(self.m1, dmdt1))

        # Advance the magnetization
        self.m1 = self.m1 + self.dt*self.dmdt1
        self.m2 = self.m2 + self.dt*self.dmdt2
        self.norm()
        self.t += self.dt
        return

    def norm(self):
        self.m1 = self.m1/np.sqrt((self.m1[0])**2 + (self.m1[1])**2 + (self.m1[2])**2)
        self.m2 = self.m2/np.sqrt((self.m2[0])**2 + (self.m2[1])**2 + (self.m2[2])**2)

    def stable(self, wait=20e-9):
        dif1 = 100
        dif2 = 100
        old1 = 100
        old2 = 100
        while dif1 > 1e-4 and dif2 > 1e-4:
            # Check the amplitude at certain phase value to update escape condition
            if self.t > wait and round(self.t/self.dt) % round(1/(self.dt*self.rfFrequency)) == \
                            round(self.phase/self.dt) % round(1/(self.dt*self.rfFrequency)):
                self.rk4()
                dif1 = np.absolute(old1 - self.m1[0])/self.m1[0]
                old1 = self.m1[0]
                dif2 = np.absolute(old2 - self.m2[0])/self.m2[0]
                old2 = self.m2[0]
            else:
                self.rk4()
            print '{}\r'.format(self.t),

    def getDynamics(self):
        # Go to start of a period and output one period of magnetization dynamics after stable condition
        while round(self.t/self.dt) % round(1/(self.dt*self.rfFrequency)) != 0:
            self.rk4()

        # Initialize arrays to output
        time = np.array([])
        mx1 = np.array([])
        my1 = np.array([])
        mz1 = np.array([])
        dmdtX1 = np.array([])
        dmdtY1 = np.array([])
        dmdtZ1 = np.array([])
        mx2 = np.array([])
        my2 = np.array([])
        mz2 = np.array([])
        dmdtX2 = np.array([])
        dmdtY2 = np.array([])
        dmdtZ2 = np.array([])

        # Step once
        self.rk4()
        time = np.append(time, self.t)
        mx1 = np.append(mx1, self.m1[0])
        my1 = np.append(my1, self.m1[1])
        mz1 = np.append(mz1, self.m1[2])
        dmdtX1 = np.append(dmdtX1, self.dmdt1[0])
        dmdtY1 = np.append(dmdtY1, self.dmdt1[1])
        dmdtZ1 = np.append(dmdtZ1, self.dmdt1[2])
        mx2 = np.append(mx2, self.m2[0])
        my2 = np.append(my2, self.m2[1])
        mz2 = np.append(mz2, self.m2[2])
        dmdtX2 = np.append(dmdtX2, self.dmdt2[0])
        dmdtY2 = np.append(dmdtY2, self.dmdt2[1])
        dmdtZ2 = np.append(dmdtZ2, self.dmdt2[2])

        # Fill in the array for 1 period
        while round(self.t/self.dt) % round(1/(self.dt*self.rfFrequency)) != 0:
            self.rk4()
            time = np.append(time, self.t)
            mx1 = np.append(mx1, self.m1[0])
            my1 = np.append(my1, self.m1[1])
            mz1 = np.append(mz1, self.m1[2])
            dmdtX1 = np.append(dmdtX1, self.dmdt1[0])
            dmdtY1 = np.append(dmdtY1, self.dmdt1[1])
            dmdtZ1 = np.append(dmdtZ1, self.dmdt1[2])
            mx2 = np.append(mx2, self.m2[0])
            my2 = np.append(my2, self.m2[1])
            mz2 = np.append(mz2, self.m2[2])
            dmdtX2 = np.append(dmdtX2, self.dmdt2[0])
            dmdtY2 = np.append(dmdtY2, self.dmdt2[1])
            dmdtZ2 = np.append(dmdtZ2, self.dmdt2[2])

        return time, mx1, my1, mz1, dmdtX1, dmdtY1, dmdtZ1, mx2, my2, mz2, dmdtX2, dmdtY2, dmdtZ2


"""
Definitions for different types of scans below.
Output is saved into a file with pickle
"""


def fieldScan(start=1, end=400, num=400, fname="fmr_4GHz_coupled_anis_sc0p0005.p", alphaSC1=0.0005, alphaSC2=0.0005):
    # Initialize arrays to save into dictionary data
    time = np.array([])
    mx1 = np.array([])
    my1 = np.array([])
    mz1 = np.array([])
    dmdtX1 = np.array([])
    dmdtY1 = np.array([])
    dmdtZ1 = np.array([])
    mx2 = np.array([])
    my2 = np.array([])
    mz2 = np.array([])
    dmdtX2 = np.array([])
    dmdtY2 = np.array([])
    dmdtZ2 = np.array([])
    Hy = np.array([])
    data = {}

    for i in np.linspace(start, end, num):
        # A new class instance at a new external field value
        coupled = FMR(m01=(0, 1, 0), m02=(0, 1, 0))
        coupled.setRF(0.1, 4e9)
        coupled.setSC(alphaSC1, alphaSC2)
        coupled.setFieldExt((0, i, 0))
        coupled.stable()

        # Update the arrays to save
        timeNew, mx1New, my1New, mz1New, dmdtX1New, dmdtY1New, dmdtZ1New, \
        mx2New, my2New, mz2New, dmdtX2New, dmdtY2New, dmdtZ2New = coupled.getDynamics()

        Hy = np.append(Hy, i)
        time = np.append(time, timeNew)
        mx1 = np.append(mx1, mx1New)
        my1 = np.append(my1, my1New)
        mz1 = np.append(mz1, mz1New)
        dmdtX1 = np.append(dmdtX1, dmdtX1New)
        dmdtY1 = np.append(dmdtY1, dmdtY1New)
        dmdtZ1 = np.append(dmdtZ1, dmdtZ1New)
        mx2 = np.append(mx2, mx2New)
        my2 = np.append(my2, my2New)
        mz2 = np.append(mz2, mz2New)
        dmdtX2 = np.append(dmdtX2, dmdtX2New)
        dmdtY2 = np.append(dmdtY2, dmdtY2New)
        dmdtZ2 = np.append(dmdtZ2, dmdtZ2New)
        print i

    data['Time'] = time
    data['Hy'] = Hy
    data['mx1'] = mx1
    data['my1'] = my1
    data['mz1'] = mz1
    data['TorqueX1'] = dmdtX1
    data['TorqueY1'] = dmdtY1
    data['TorqueZ1'] = dmdtZ1
    data['Hy'] = Hy
    data['mx2'] = mx2
    data['my2'] = my2
    data['mz2'] = mz2
    data['TorqueX2'] = dmdtX2
    data['TorqueY2'] = dmdtY2
    data['TorqueZ2'] = dmdtZ2

    data['FieldPoints'] = num
    data['TimePoints'] = 1/(coupled.rfFrequency*coupled.dt)
    data['RF Frequency'] = coupled.rfFrequency

    pickle.dump(data, open(fname, "wb"))


def timeScan(Hy=200):
    dict = {}
    num = 201
    step = 2.5e-12

    time = np.zeros(num)
    mx = np.zeros(num)
    my = np.zeros(num)
    mz = np.zeros(num)

    for i in range(0, num):
        phase = i*step
        time[i] = phase
        py = FMR(m0=(0,1,0))
        py.setRF(0.1, 4e9)
        py.setFieldExt((0, Hy, 0))
        py.setPhase(phase=phase)
        py.stable()
        print py.phase, py.m
        mx[i] = py.m[0]
        my[i] = py.m[1]
        mz[i] = py.m[2]

    phaseDif = np.arccos(mx[0]/np.max(mx))

    dict['Time'] = time
    dict['mx'] = mx
    dict['my'] = my
    dict['mz'] = mz
    dict['phaseDif'] = phaseDif

    return dict


def dynamics():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    py = FMR(m0=(0, 1, 0))
    py.setRF(0.1, 4e9)
    py.setFieldExt((0, 200, 0))
    py.setPhase(phase=62.5e-12)
    py.stable()
    data = py.getDynamics()
    ax.plot(data['Time'], data['mx'], 'rs', ms=6)
    plt.show()


def phaseScan():
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.set_ylabel('mx')
    ax2 = fig.add_subplot(212)
    ax2.set_ylabel('phase')
    ax2.set_xlabel('Hy (Oe)')
    num = 3
    field = np.zeros(num)
    phase = np.zeros(num)
    node = np.zeros(num)
    antinode = np.zeros(num)
    data = {}
    for i in range(0, num):
        field[i] = i*200
        temp = timeScan(Hy=i*200)
        phase[i] = temp['phaseDif']
        node[i] = temp['mx'][25]
        antinode[i] = temp['mx'][0]
        print field[i]

    #data = pickle.load(open("save.p", "rb"))
    ax1.plot(field, node, 'rs', field, antinode, 'g^', ms=6, label='resonance')
    ax2.plot(field, phase, 'bo', ms=6, label='phase')
    data['Field'] = field
    data['node'] = node
    data['antinode'] = antinode
    data['phase'] = phase

    pickle.dump(data, open("fmr_py.p", "wb"))
    #print pyDict
    #print "Phase Difference =", data['phaseDif']
    plt.show()


def test():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    py = FMR(m0=(0, 1, 0))
    py.setRF(0.1, 4e9)
    py.setFieldExt((0, 200, 0))
    py.setPhase(phase=62.5e-12)

    for i in range(0,10000):
        #py.dt = 1e-12
        py.rk4()
        print '{}\r'.format(i),
        ax.plot(py.t, py.m[1], 'g^', ms=6)
    plt.show()


"""
Definitions to load the desired data from the saved file above
"""


def load():
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    data = pickle.load(open("fmr_4GHz_torque2.p", "rb"))
    Hy = data['Hy']
    print data['mx'].shape
    num = 400
    node = np.zeros(num + 1)
    antinode = np.zeros(num + 1)
    phase = np.zeros(num + 1)
    for i in range(0, num):
        #time = np.reshape(data['Time'], (num + 1, 2500))[i, :]
        node[i] = np.reshape(data['mx'], (num + 1, 2500))[i, 0]
        antinode[i] = np.reshape(data['mx'], (num + 1, 2500))[i, 624]
        phase[i] = np.arccos(np.reshape(data['mx'], (num + 1, 2500))[i, 0]
                             /np.max(np.reshape(data['mx'], (num + 1, 2500))[i, :]))
    """
    phase = np.zeros(len(data))
    for i in range(0, len(data)-1):
        Hy[i] = i*2
        mx[i] = data[i]['mx'][624]
        phase[i] = np.arccos(data[i]['mx'][0]/np.max(data[i]['mx']))
    """
    ax1.plot(Hy, node, 'rs', Hy, antinode, 'g^', ms=6)
    ax2.plot(Hy, phase, 'bo', ms=6)
    plt.show()


def loadTimeScan(Hy=200, fname="D:\\PyCharm\\FMR\\fmr_4GHz_noSpinCurrent.p"):
    fig = plt.figure()
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)
    data = pickle.load(open(fname, "rb"))
    num = data['FieldPoints']
    numTime = data['TimePoints']
    time = np.reshape(data['Time'], (num, numTime))[Hy, :]

    mx1 = np.reshape(data['mx1'], (num, numTime))[Hy, :]
    my1 = np.reshape(data['my1'], (num, numTime))[Hy, :]
    mz1 = np.reshape(data['mz1'], (num, numTime))[Hy, :]
    mx2 = np.reshape(data['mx2'], (num, numTime))[Hy, :]
    my2 = np.reshape(data['my2'], (num, numTime))[Hy, :]
    mz2 = np.reshape(data['mz2'], (num, numTime))[Hy, :]
    torque = np.zeros(num)

    ax1.plot(time, mx1, 'r-', ms=6)
    ax2.plot(time, my1, 'b-', ms=6)
    ax3.plot(time, mz1, 'g-', ms=6)
    ax4.plot(time, mx2, 'r-', ms=6)
    ax5.plot(time, my2, 'b-', ms=6)
    ax6.plot(time, mz2, 'g-', ms=6)
    plt.show()


def loadFieldScan(t=624, fname="D:\\PyCharm\\FMR\\fmr_4GHz_alphaSC_0p003.p"):
    # Prepare figure
    fig = plt.figure(figsize=(12, 8), dpi=100)
    ax1 = fig.add_subplot(211)
    #ax2 = ax1.twinx()
    ax3 = fig.add_subplot(212)
    #ax4 = ax3.twinx()

    # Load data using pickle
    data = pickle.load(open(fname, "rb"))
    num = data['FieldPoints']
    numTime = data['TimePoints']

    # Initialize some arrays to be plotted
    mx1 = []
    my1 = []
    mz1 = []
    dmdtX1 = []
    dmdtY1 = []
    dmdtZ1 = []
    phase1 = []
    mx2 = []
    my2 = []
    mz2 = []
    dmdtX2 = []
    dmdtY2 = []
    dmdtZ2 = []
    phase2 = []
    Hy = data['Hy']

    for i in range(0, 300):
        # Update layer 1 properties
        mx1 = np.append(mx1, np.reshape(data['mx1'], (num, numTime))[i, :][t])
        my1 = np.append(my1, np.reshape(data['my1'], (num, numTime))[i, :][t])
        mz1 = np.append(mz1, np.reshape(data['mz1'], (num, numTime))[i, :][t])
        dmdtX1 = np.append(dmdtX1, np.reshape(data['TorqueX1'], (num, numTime))[i, :][t])
        dmdtY1 = np.append(dmdtY1, np.reshape(data['TorqueY1'], (num, numTime))[i, :][t])
        dmdtZ1 = np.append(dmdtZ1, np.reshape(data['TorqueZ1'], (num, numTime))[i, :][t])
        phase1 = np.append(phase1, np.arccos(
            np.reshape(data['mx1'], (num, numTime))[i, 0] /
            np.max(np.reshape(data['mx1'], (num, numTime))[i, :])))

        # Update layer 2 properties
        mx2 = np.append(mx2, np.reshape(data['mx2'], (num, numTime))[i, :][t])
        my2 = np.append(my2, np.reshape(data['my2'], (num, numTime))[i, :][t])
        mz2 = np.append(mz2, np.reshape(data['mz2'], (num, numTime))[i, :][t])
        dmdtX2 = np.append(dmdtX2, np.reshape(data['TorqueX2'], (num, numTime))[i, :][t])
        dmdtY2 = np.append(dmdtY2, np.reshape(data['TorqueY2'], (num, numTime))[i, :][t])
        dmdtZ2 = np.append(dmdtZ2, np.reshape(data['TorqueZ2'], (num, numTime))[i, :][t])
        phase2 = np.append(phase2, np.arccos(
            np.reshape(data['mx2'], (num, numTime))[i, 0] /
            np.max(np.reshape(data['mx2'], (num, numTime))[i, :])))

    ax1.plot(Hy[0:300], mx1, 'r-', ms=6, label='Py')
    ax1.plot(Hy[0:300], mx2, 'b-', ms=6, label='Co')
    ax3.plot(Hy[0:300], phase1, 'r-', ms=6, label='Py')
    ax3.plot(Hy[0:300], phase2, 'b-', ms=6, label='Py')
    ax1.set_xlabel('Field (Oe)')
    ax3.set_xlabel('Field (Oe)')
    plt.legend(loc='upper right')
    ax1.set_ylabel('Amplitude (a.u.)')
    ax3.set_ylabel('Phase')
    #ax3.legend(loc='upper right')
    plt.show()
    return mx1, mx2, phase1, phase2, Hy


def loadSC(Hy=200):
    fig = plt.figure()
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)
    data = pickle.load(open("fmr_4GHz_alphaSC_0p003.p", "rb"))
    num = data['FieldPoints']
    numTime = data['TimePoints']
    time = np.reshape(data['Time'], (num, numTime))[Hy, :]

    a = np.reshape(data['mx1'], (num, numTime))[Hy, :]
    b = np.reshape(data['my1'], (num, numTime))[Hy, :]
    c = np.reshape(data['mz1'], (num, numTime))[Hy, :]
    m1 = np.concatenate((a[..., np.newaxis],
                        b[..., np.newaxis],
                        c[..., np.newaxis]), axis=1)

    a = np.reshape(data['mx2'], (num, numTime))[Hy, :]
    b = np.reshape(data['my2'], (num, numTime))[Hy, :]
    c = np.reshape(data['mz2'], (num, numTime))[Hy, :]
    m2 = np.concatenate((a[..., np.newaxis],
                        b[..., np.newaxis],
                        c[..., np.newaxis]), axis=1)

    a = np.reshape(data['TorqueX1'], (num, numTime))[Hy, :]
    b = np.reshape(data['TorqueY1'], (num, numTime))[Hy, :]
    c = np.reshape(data['TorqueZ1'], (num, numTime))[Hy, :]
    dmdt1 = np.concatenate((a[..., np.newaxis],
                        b[..., np.newaxis],
                        c[..., np.newaxis]), axis=1)

    a = np.reshape(data['TorqueX2'], (num, numTime))[Hy, :]
    b = np.reshape(data['TorqueY2'], (num, numTime))[Hy, :]
    c = np.reshape(data['TorqueZ2'], (num, numTime))[Hy, :]
    dmdt2 = np.concatenate((a[..., np.newaxis],
                        b[..., np.newaxis],
                        c[..., np.newaxis]), axis=1)

    sc1 = -0.003*np.cross(m1, dmdt1)
    sc2 = 0.003*np.cross(m2, dmdt2)

    ax1.plot(time, m1[:, 0], 'r-', ms=6)
    ax2.plot(time, m1[:, 1], 'b-', ms=6)
    ax3.plot(time, m1[:, 2], 'g-', ms=6)
    ax4.plot(time, sc1[:, 0], 'r-', ms=6)
    ax5.plot(time, sc1[:, 1], 'b-', ms=6)
    ax6.plot(time, sc1[:, 2], 'g-', ms=6)
    plt.show()

#loadFieldScan(fname="D:\\Dropbox\\PyCharm\\FMR\\fmr_4GHz_coupled_anis_noSC.p")
