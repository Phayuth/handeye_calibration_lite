import math
import numpy as np

X_AXIS = np.array([1.0, 0.0, 0.0], dtype=np.float64)
Y_AXIS = np.array([0.0, 1.0, 0.0], dtype=np.float64)
Z_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)
# epsilon for testing whether a number is close to zero
_FLOAT_EPS = np.finfo(float).eps
_EPS = np.finfo(float).eps * 4.0
# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]
# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    "sxyz": (0, 0, 0, 0),
    "sxyx": (0, 0, 1, 0),
    "sxzy": (0, 1, 0, 0),
    "sxzx": (0, 1, 1, 0),
    "syzx": (1, 0, 0, 0),
    "syzy": (1, 0, 1, 0),
    "syxz": (1, 1, 0, 0),
    "syxy": (1, 1, 1, 0),
    "szxy": (2, 0, 0, 0),
    "szxz": (2, 0, 1, 0),
    "szyx": (2, 1, 0, 0),
    "szyz": (2, 1, 1, 0),
    "rzyx": (0, 0, 0, 1),
    "rxyx": (0, 0, 1, 1),
    "ryzx": (0, 1, 0, 1),
    "rxzx": (0, 1, 1, 1),
    "rxzy": (1, 0, 0, 1),
    "ryzy": (1, 0, 1, 1),
    "rzxy": (1, 1, 0, 1),
    "ryxy": (1, 1, 1, 1),
    "ryxz": (2, 0, 0, 1),
    "rzxz": (2, 0, 1, 1),
    "rxyz": (2, 1, 0, 1),
    "rzyz": (2, 1, 1, 1),
}
_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


class br_axis_angle:

    @staticmethod
    def to_euler(axis, angle, axes="sxyz"):
        """
        Return Euler angles from a rotation in the axis-angle representation.

        Parameters
        ----------
        axis: array_like
            axis around which the rotation occurs
        angle: float
            angle of rotation
        axes: str, optional
            Axis specification; one of 24 axis sequences as string or encoded tuple

        Returns
        -------
        ai: float
            First rotation angle (according to axes).
        aj: float
            Second rotation angle (according to axes).
        ak: float
            Third rotation angle (according to axes).
        """
        T = br_axis_angle.to_transform(axis, angle)
        return br_transform.to_euler(T, axes)

    @staticmethod
    def to_quaternion(axis, angle, isunit=False):
        """
        Return quaternion from a rotation in the axis-angle representation.

        Parameters
        ----------
        axis: array_like
            axis around which the rotation occurs
        angle: float
            angle of rotation

        Returns
        -------
        q: array_like
            Quaternion in w, x, y z (real, then vector) format

        Notes
        -----
        Quaternions :math:`w + ix + jy + kz` are represented as :math:`[w, x, y, z]`.
        """
        u = np.array(axis)
        if not isunit:
            # Cannot divide in-place because input vector may be integer type,
            # whereas output will be float type; this may raise an error in versions
            # of numpy > 1.6.1
            u = u / math.sqrt(np.dot(u, u))
        t2 = angle / 2.0
        st2 = math.sin(t2)
        return np.concatenate(([math.cos(t2)], u * st2))

    @staticmethod
    def to_transform(axis, angle, point=None):
        """
        Return homogeneous transformation from an axis-angle rotation.

        Parameters
        ----------
        axis: array_like
            axis around which the rotation occurs
        angle: float
            angle of rotation
        point: array_like
            point around which the rotation is performed

        Returns
        -------
        T: array_like
            Homogeneous transformation (4x4)
        """
        sina = math.sin(angle)
        cosa = math.cos(angle)
        axis = br_vector.unit(axis[:3])
        # rotation matrix around unit vector
        R = np.diag([cosa, cosa, cosa])
        R += np.outer(axis, axis) * (1.0 - cosa)
        axis *= sina
        R += np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ]
        )
        T = np.identity(4)
        T[:3, :3] = R
        if point is not None:
            # rotation not around origin
            point = np.array(point[:3], dtype=np.float64, copy=False)
            T[:3, 3] = point - np.dot(R, point)
        return T


class br_euler:
    @staticmethod
    def to_axis_angle(ai, aj, ak, axes="sxyz"):
        """
        Return axis-angle rotation from Euler angles and axes sequence

        Parameters
        ----------
        ai: float
            First rotation angle (according to axes).
        aj: float
            Second rotation angle (according to axes).
        ak: float
            Third rotation angle (according to axes).
        axes: str, optional
            Axis specification; one of 24 axis sequences as string or encoded tuple

        Returns
        -------
        axis: array_like
            axis around which rotation occurs
        angle: float
            angle of rotation

        Examples
        --------
        >>> import numpy as np
        >>> import baldor as br
        >>> axis, angle = br.euler.to_axis_angle(0, 1.5, 0, 'szyx')
        >>> np.allclose(axis, [0, 1, 0])
        True
        >>> angle
        1.5
        """
        T = br_euler.to_transform(ai, aj, ak, axes)
        axis, angle, _ = br_transform.to_axis_angle(T)
        return axis, angle

    @staticmethod
    def to_quaternion(ai, aj, ak, axes="sxyz"):
        """
        Returns a quaternion from Euler angles and axes sequence

        Parameters
        ----------
        ai: float
            First rotation angle (according to axes).
        aj: float
            Second rotation angle (according to axes).
        ak: float
            Third rotation angle (according to axes).
        axes: str, optional
            Axis specification; one of 24 axis sequences as string or encoded tuple

        Returns
        -------
        q: array_like
            Quaternion in w, x, y z (real, then vector) format

        Notes
        -----
        Quaternions :math:`w + ix + jy + kz` are represented as :math:`[w, x, y, z]`.

        Examples
        --------
        >>> import numpy as np
        >>> import baldor as br
        >>> q = br.euler.to_quaternion(1, 2, 3, 'ryxz')
        >>> np.allclose(q, [0.435953, 0.310622, -0.718287, 0.444435])
        True
        """
        try:
            firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
        except (AttributeError, KeyError):
            _TUPLE2AXES[axes]  # validation
            firstaxis, parity, repetition, frame = axes

        i = firstaxis + 1
        j = _NEXT_AXIS[i + parity - 1] + 1
        k = _NEXT_AXIS[i - parity] + 1

        if frame:
            ai, ak = ak, ai
        if parity:
            aj = -aj

        ai /= 2.0
        aj /= 2.0
        ak /= 2.0
        ci = math.cos(ai)
        si = math.sin(ai)
        cj = math.cos(aj)
        sj = math.sin(aj)
        ck = math.cos(ak)
        sk = math.sin(ak)
        cc = ci * ck
        cs = ci * sk
        sc = si * ck
        ss = si * sk

        q = np.empty((4,))
        if repetition:
            q[0] = cj * (cc - ss)
            q[i] = cj * (cs + sc)
            q[j] = sj * (cc + ss)
            q[k] = sj * (cs - sc)
        else:
            q[0] = cj * cc + sj * ss
            q[i] = cj * sc - sj * cs
            q[j] = cj * ss + sj * cc
            q[k] = cj * cs - sj * sc
        if parity:
            q[j] *= -1.0
        return q

    @staticmethod
    def to_transform(ai, aj, ak, axes="sxyz"):
        """
        Return homogeneous transformation matrix from Euler angles and axes sequence.

        Parameters
        ----------
        ai: float
            First rotation angle (according to axes).
        aj: float
            Second rotation angle (according to axes).
        ak: float
            Third rotation angle (according to axes).
        axes: str, optional
            Axis specification; one of 24 axis sequences as string or encoded tuple

        Returns
        -------
        T: array_like
            Homogeneous transformation (4x4)

        Examples
        --------
        >>> import numpy as np
        >>> import baldor as br
        >>> T = br.euler.to_transform(1, 2, 3, 'syxz')
        >>> np.allclose(np.sum(T[0]), -1.34786452)
        True
        >>> T = br.euler.to_transform(1, 2, 3, (0, 1, 0, 1))
        >>> np.allclose(np.sum(T[0]), -0.383436184)
        True
        """
        try:
            firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
        except (AttributeError, KeyError):
            _TUPLE2AXES[axes]  # validation
            firstaxis, parity, repetition, frame = axes

        i = firstaxis
        j = _NEXT_AXIS[i + parity]
        k = _NEXT_AXIS[i - parity + 1]

        if frame:
            ai, ak = ak, ai
        if parity:
            ai, aj, ak = -ai, -aj, -ak

        si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
        ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
        cc, cs = ci * ck, ci * sk
        sc, ss = si * ck, si * sk

        T = np.identity(4)
        if repetition:
            T[i, i] = cj
            T[i, j] = sj * si
            T[i, k] = sj * ci
            T[j, i] = sj * sk
            T[j, j] = -cj * ss + cc
            T[j, k] = -cj * cs - sc
            T[k, i] = -sj * ck
            T[k, j] = cj * sc + cs
            T[k, k] = cj * cc - ss
        else:
            T[i, i] = cj * ck
            T[i, j] = sj * sc - cs
            T[i, k] = sj * cc + ss
            T[j, i] = cj * sk
            T[j, j] = sj * ss + cc
            T[j, k] = sj * cs - sc
            T[k, i] = -sj
            T[k, j] = cj * si
            T[k, k] = cj * ci
        return T


class br_quaternion:
    @staticmethod
    def are_equal(q1, q2, rtol=1e-5, atol=1e-8):
        """
        Returns True if two quaternions are equal within a tolerance.

        Parameters
        ----------
        q1: array_like
            First input quaternion (4 element sequence)
        q2: array_like
            Second input quaternion (4 element sequence)
        rtol: float
            The relative tolerance parameter.
        atol: float
            The absolute tolerance parameter.

        Returns
        -------
        equal : bool
            True if `q1` and `q2` are `almost` equal, False otherwise

        See Also
        --------
        numpy.allclose: Contains the details about the tolerance parameters

        Notes
        -----
        Quaternions :math:`w + ix + jy + kz` are represented as :math:`[w, x, y, z]`.

        Examples
        --------
        >>> import baldor as br
        >>> q1 = [1, 0, 0, 0]
        >>> br.quaternion.are_equal(q1, [0, 1, 0, 0])
        False
        >>> br.quaternion.are_equal(q1, [1, 0, 0, 0])
        True
        >>> br.quaternion.are_equal(q1, [-1, 0, 0, 0])
        True
        """
        if np.allclose(q1, q2, rtol, atol):
            return True
        return np.allclose(np.array(q1) * -1, q2, rtol, atol)

    @staticmethod
    def conjugate(q):
        """
        Compute the conjugate of a quaternion.

        Parameters
        ----------
        q: array_like
            Input quaternion (4 element sequence)

        Returns
        -------
        qconj: ndarray
            The conjugate of the input quaternion.

        Notes
        -----
        Quaternions :math:`w + ix + jy + kz` are represented as :math:`[w, x, y, z]`.

        Examples
        --------
        >>> import baldor as br
        >>> q0 = br.quaternion.random()
        >>> q1 = br.quaternion.conjugate(q0)
        >>> q1[0] == q0[0] and all(q1[1:] == -q0[1:])
        True

        """
        qconj = np.array(q, dtype=np.float64, copy=True)
        np.negative(qconj[1:], qconj[1:])
        return qconj

    @staticmethod
    def dual_to_transform(qr, qt):
        """
        Return a homogeneous transformation from the given dual quaternion.

        Parameters
        ----------
        qr: array_like
            Input quaternion for the rotation component (4 element sequence)
        qt: array_like
            Input quaternion for the translation component (4 element sequence)

        Returns
        -------
        T: array_like
            Homogeneous transformation (4x4)

        Notes
        -----
        Some literature prefers to use :math:`q` for the rotation component and
        :math:`q'` for the translation component
        """
        T = np.eye(4)
        R = br_quaternion.to_transform(qr)[:3, :3]
        t = 2 * br_quaternion.multiply(qt, br_quaternion.conjugate(qr))
        T[:3, :3] = R
        T[:3, 3] = t[1:]
        return T

    @staticmethod
    def inverse(q):
        """
        Return multiplicative inverse of a quaternion

        Parameters
        ----------
        q: array_like
            Input quaternion (4 element sequence)

        Returns
        -------
        qinv : ndarray
            The inverse of the input quaternion.

        Notes
        -----
        Quaternions :math:`w + ix + jy + kz` are represented as :math:`[w, x, y, z]`.
        """
        return br_quaternion.conjugate(q) / br_quaternion.norm(q)

    @staticmethod
    def multiply(q1, q2):
        """
        Multiply two quaternions

        Parameters
        ----------
        q1: array_like
            First input quaternion (4 element sequence)
        q2: array_like
            Second input quaternion (4 element sequence)

        Returns
        -------
        result: ndarray
            The resulting quaternion

        Notes
        -----
        `Hamilton product of quaternions
        <http://en.wikipedia.org/wiki/Quaternions#Hamilton_product>`_

        Quaternions :math:`w + ix + jy + kz` are represented as :math:`[w, x, y, z]`.

        Examples
        --------
        >>> import numpy as np
        >>> import baldor as br
        >>> q = br.quaternion.multiply([4, 1, -2, 3], [8, -5, 6, 7])
        >>> np.allclose(q, [28, -44, -14, 48])
        True
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array(
            [
                -x1 * x2 - y1 * y2 - z1 * z2 + w1 * w2,
                x1 * w2 + y1 * z2 - z1 * y2 + w1 * x2,
                -x1 * z2 + y1 * w2 + z1 * x2 + w1 * y2,
                x1 * y2 - y1 * x2 + z1 * w2 + w1 * z2,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def norm(q):
        """
        Compute quaternion norm

        Parameters
        ----------
        q : array_like
            Input quaternion (4 element sequence)

        Returns
        -------
        n : float
            quaternion norm

        Notes
        -----
        Quaternions :math:`w + ix + jy + kz` are represented as :math:`[w, x, y, z]`.
        """
        return np.dot(q, q)

    @staticmethod
    def random(rand=None):
        """
        Generate an uniform random unit quaternion.

        Parameters
        ----------
        rand: array_like or None
            Three independent random variables that are uniformly distributed
            between 0 and 1.

        Returns
        -------
        qrand: array_like
            The random quaternion

        Notes
        -----
        Quaternions :math:`w + ix + jy + kz` are represented as :math:`[w, x, y, z]`.

        Examples
        --------
        >>> import numpy as np
        >>> import baldor as br
        >>> q = br.quaternion.random()
        >>> np.allclose(1, np.linalg.norm(q))
        True
        """
        if rand is None:
            rand = np.random.rand(3)
        else:
            assert len(rand) == 3
        r1 = np.sqrt(1.0 - rand[0])
        r2 = np.sqrt(rand[0])
        pi2 = math.pi * 2.0
        t1 = pi2 * rand[1]
        t2 = pi2 * rand[2]
        return np.array(
            [np.cos(t2) * r2, np.sin(t1) * r1, np.cos(t1) * r1, np.sin(t2) * r2]
        )

    @staticmethod
    def to_axis_angle(quaternion, identity_thresh=None):
        """
        Return axis-angle rotation from a quaternion

        Parameters
        ----------
        quaternion: array_like
            Input quaternion (4 element sequence)
        identity_thresh : None or scalar, optional
            Threshold below which the norm of the vector part of the quaternion (x,
            y, z) is deemed to be 0, leading to the identity rotation.  None (the
            default) leads to a threshold estimated based on the precision of the
            input.

        Returns
        ----------
        axis: array_like
            axis around which rotation occurs
        angle: float
            angle of rotation

        Notes
        -----
        Quaternions :math:`w + ix + jy + kz` are represented as :math:`[w, x, y, z]`.
        A quaternion for which x, y, z are all equal to 0, is an identity rotation.
        In this case we return a `angle=0` and `axis=[1, 0, 0]``. This is an arbitrary
        vector.

        Examples
        --------
        >>> import numpy as np
        >>> import baldor as br
        >>> axis, angle = br.euler.to_axis_angle(0, 1.5, 0, 'szyx')
        >>> np.allclose(axis, [0, 1, 0])
        True
        >>> angle
        1.5
        """
        w, x, y, z = quaternion
        Nq = br_quaternion.norm(quaternion)
        if not np.isfinite(Nq):
            return np.array([1.0, 0, 0]), float("nan")
        if identity_thresh is None:
            try:
                identity_thresh = np.finfo(Nq.type).eps * 3
            except (AttributeError, ValueError):  # Not a numpy type or not float
                identity_thresh = _FLOAT_EPS * 3
        if Nq < _FLOAT_EPS**2:  # Results unreliable after normalization
            return np.array([1.0, 0, 0]), 0.0
        if not np.isclose(Nq, 1):  # Normalize if not normalized
            s = math.sqrt(Nq)
            w, x, y, z = w / s, x / s, y / s, z / s
        len2 = x * x + y * y + z * z
        if len2 < identity_thresh**2:
            # if vec is nearly 0,0,0, this is an identity rotation
            return np.array([1.0, 0, 0]), 0.0
        # Make sure w is not slightly above 1 or below -1
        theta = 2 * math.acos(max(min(w, 1), -1))
        return np.array([x, y, z]) / math.sqrt(len2), theta

    @staticmethod
    def to_euler(quaternion, axes="sxyz"):
        """
        Return Euler angles from a quaternion using the specified axis sequence.

        Parameters
        ----------
        q : array_like
            Input quaternion (4 element sequence)
        axes: str, optional
            Axis specification; one of 24 axis sequences as string or encoded tuple

        Returns
        -------
        ai: float
            First rotation angle (according to axes).
        aj: float
            Second rotation angle (according to axes).
        ak: float
            Third rotation angle (according to axes).

        Notes
        -----
        Many Euler angle triplets can describe the same rotation matrix
        Quaternions :math:`w + ix + jy + kz` are represented as :math:`[w, x, y, z]`.

        Examples
        --------
        >>> import numpy as np
        >>> import baldor as br
        >>> ai, aj, ak = br.quaternion.to_euler([0.99810947, 0.06146124, 0, 0])
        >>> np.allclose([ai, aj, ak], [0.123, 0, 0])
        True
        """
        return br_transform.to_euler(br_quaternion.to_transform(quaternion), axes)

    @staticmethod
    def to_transform(quaternion):
        """
        Return homogeneous transformation from a quaternion.

        Parameters
        ----------
        quaternion: array_like
            Input quaternion (4 element sequence)
        axes: str, optional
            Axis specification; one of 24 axis sequences as string or encoded tuple

        Returns
        -------
        T: array_like
            Homogeneous transformation (4x4)

        Notes
        -----
        Quaternions :math:`w + ix + jy + kz` are represented as :math:`[w, x, y, z]`.

        Examples
        --------
        >>> import numpy as np
        >>> import baldor as br
        >>> T0 = br.quaternion.to_transform([1, 0, 0, 0]) # Identity quaternion
        >>> np.allclose(T0, np.eye(4))
        True
        >>> T1 = br.quaternion.to_transform([0, 1, 0, 0]) # 180 degree rot around X
        >>> np.allclose(T1, np.diag([1, -1, -1, 1]))
        True
        """
        q = np.array(quaternion, dtype=np.float64, copy=True)
        n = np.dot(q, q)
        if n < _EPS:
            return np.identity(4)
        q *= math.sqrt(2.0 / n)
        q = np.outer(q, q)
        return np.array(
            [
                [
                    1.0 - q[2, 2] - q[3, 3],
                    q[1, 2] - q[3, 0],
                    q[1, 3] + q[2, 0],
                    0.0,
                ],
                [
                    q[1, 2] + q[3, 0],
                    1.0 - q[1, 1] - q[3, 3],
                    q[2, 3] - q[1, 0],
                    0.0,
                ],
                [
                    q[1, 3] - q[2, 0],
                    q[2, 3] + q[1, 0],
                    1.0 - q[1, 1] - q[2, 2],
                    0.0,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )


class br_transform:
    @staticmethod
    def are_equal(T1, T2, rtol=1e-5, atol=1e-8):
        """
        Returns True if two homogeneous transformation are equal within a tolerance.

        Parameters
        ----------
        T1: array_like
            First input homogeneous transformation
        T2: array_like
            Second input homogeneous transformation
        rtol: float
            The relative tolerance parameter.
        atol: float
            The absolute tolerance parameter.

        Returns
        -------
        equal : bool
            True if `T1` and `T2` are `almost` equal, False otherwise

        See Also
        --------
        numpy.allclose: Contains the details about the tolerance parameters
        """
        M1 = np.array(T1, dtype=np.float64, copy=True)
        M1 /= M1[3, 3]
        M2 = np.array(T2, dtype=np.float64, copy=True)
        M2 /= M2[3, 3]
        return np.allclose(M1, M2, rtol, atol)

    @staticmethod
    def between_axes(axis_a, axis_b):
        """
        Compute the transformation that aligns two vectors/axes.

        Parameters
        ----------
        axis_a: array_like
            The initial axis
        axis_b: array_like
            The goal axis

        Returns
        -------
        transform: array_like
            The transformation that transforms `axis_a` into `axis_b`
        """
        a_unit = br_vector.unit(axis_a)
        b_unit = br_vector.unit(axis_b)
        c = np.dot(a_unit, b_unit)
        angle = np.arccos(c)
        if np.isclose(c, -1.0) or np.allclose(a_unit, b_unit):
            axis = br_vector.perpendicular(b_unit)
        else:
            axis = br_vector.unit(np.cross(a_unit, b_unit))
        transform = br_axis_angle.to_transform(axis, angle)
        return transform

    @staticmethod
    def inverse(transform):
        """
        Compute the inverse of an homogeneous transformation.

        .. note:: This function is more efficient than :obj:`numpy.linalg.inv` given
            the special properties of homogeneous transformations.

        Parameters
        ----------
        transform: array_like
            The input homogeneous transformation

        Returns
        -------
        inv: array_like
            The inverse of the input homogeneous transformation
        """
        R = transform[:3, :3].T
        p = transform[:3, 3]
        inv = np.eye(4)
        inv[:3, :3] = R
        inv[:3, 3] = np.dot(-R, p)
        return inv

    @staticmethod
    def random(max_position=1.0):
        """
        Generate a random homogeneous transformation.

        Parameters
        ----------
        max_position: float, optional
            Maximum value for the position components of the transformation

        Returns
        -------
        T: array_like
            The random homogeneous transformation

        Examples
        --------
        >>> import numpy as np
        >>> import baldor as br
        >>> T = br.transform.random()
        >>> Tinv = br.transform.inverse(T)
        >>> np.allclose(np.dot(T, Tinv), np.eye(4))
        True
        """
        quat = br_quaternion.random()
        T = br_quaternion.to_transform(quat)
        T[:3, 3] = np.random.rand(3) * max_position
        return T

    @staticmethod
    def to_axis_angle(transform):
        """
        Return rotation angle and axis from rotation matrix.

        Parameters
        ----------
        transform: array_like
            The input homogeneous transformation

        Returns
        -------
        axis: array_like
            axis around which rotation occurs
        angle: float
            angle of rotation
        point: array_like
            point around which the rotation is performed

        Examples
        --------
        >>> import numpy as np
        >>> import baldor as br
        >>> axis = np.random.sample(3) - 0.5
        >>> angle = (np.random.sample(1) - 0.5) * (2*np.pi)
        >>> point = np.random.sample(3) - 0.5
        >>> T0 = br.axis_angle.to_transform(axis, angle, point)
        >>> axis, angle, point = br.transform.to_axis_angle(T0)
        >>> T1 = br.axis_angle.to_transform(axis, angle, point)
        >>> br.transform.are_equal(T0, T1)
        True
        """
        R = np.array(transform, dtype=np.float64, copy=False)
        R33 = R[:3, :3]
        # direction: unit eigenvector of R33 corresponding to eigenvalue of 1
        w, W = np.linalg.eig(R33.T)
        i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
        if not len(i):
            raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
        axis = np.real(W[:, i[-1]]).squeeze()
        # point: unit eigenvector of R corresponding to eigenvalue of 1
        w, Q = np.linalg.eig(R)
        i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
        if not len(i):
            raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
        point = np.real(Q[:, i[-1]]).squeeze()
        point /= point[3]
        # rotation angle depending on axis
        cosa = (np.trace(R33) - 1.0) / 2.0
        if abs(axis[2]) > 1e-8:
            sina = (R[1, 0] + (cosa - 1.0) * axis[0] * axis[1]) / axis[2]
        elif abs(axis[1]) > 1e-8:
            sina = (R[0, 2] + (cosa - 1.0) * axis[0] * axis[2]) / axis[1]
        else:
            sina = (R[2, 1] + (cosa - 1.0) * axis[1] * axis[2]) / axis[0]
        angle = math.atan2(sina, cosa)
        return axis, angle, point

    @staticmethod
    def to_dual_quaternion(transform):
        """
        Return quaternion from the rotation part of an homogeneous transformation.

        Parameters
        ----------
        transform: array_like
            Rotation matrix. It can be (3x3) or (4x4)
        isprecise: bool
            If True, the input transform is assumed to be a precise rotation matrix and
            a faster algorithm is used.

        Returns
        -------
        qr: array_like
            Quaternion in w, x, y z (real, then vector) for the rotation component
        qt: array_like
            Quaternion in w, x, y z (real, then vector) for the translation component

        Notes
        -----
        Some literature prefers to use :math:`q` for the rotation component and
        :math:`q'` for the translation component
        """
        cot = lambda x: 1.0 / np.tan(x)
        R = np.eye(4)
        R[:3, :3] = transform[:3, :3]
        l, theta, _ = br_transform.to_axis_angle(R)
        t = transform[:3, 3]
        # Pitch d
        d = np.dot(l.reshape(1, 3), t.reshape(3, 1))
        # Point c
        c = 0.5 * (t - d * l) + cot(theta / 2.0) * np.cross(l, t)
        # Moment vector
        m = np.cross(c, l)
        # Rotation quaternion
        qr = np.zeros(4)
        qr[0] = np.cos(theta / 2.0)
        qr[1:] = np.sin(theta / 2.0) * l
        # Translation quaternion
        qt = np.zeros(4)
        qt[0] = -(1 / 2.0) * np.dot(qr[1:], t)
        qt[1:] = (1 / 2.0) * (qr[0] * t + np.cross(t, qr[1:]))
        return qr, qt

    @staticmethod
    def to_euler(transform, axes="sxyz"):
        """
        Return Euler angles from transformation matrix with the specified axis
        sequence.

        Parameters
        ----------
        transform: array_like
            Rotation matrix. It can be (3x3) or (4x4)
        axes: str, optional
            Axis specification; one of 24 axis sequences as string or encoded tuple

        Returns
        -------
        ai: float
            First rotation angle (according to axes).
        aj: float
            Second rotation angle (according to axes).
        ak: float
            Third rotation angle (according to axes).

        Notes
        -----
        Many Euler angle triplets can describe the same rotation matrix

        Examples
        --------
        >>> import numpy as np
        >>> import baldor as br
        >>> T0 = br.euler.to_transform(1, 2, 3, 'syxz')
        >>> al, be, ga = br.transform.to_euler(T0, 'syxz')
        >>> T1 = br.euler.to_transform(al, be, ga, 'syxz')
        >>> np.allclose(T0, T1)
        True
        """
        try:
            firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
        except (AttributeError, KeyError):
            _TUPLE2AXES[axes]  # validation
            firstaxis, parity, repetition, frame = axes

        i = firstaxis
        j = _NEXT_AXIS[i + parity]
        k = _NEXT_AXIS[i - parity + 1]

        M = np.array(transform, dtype=np.float64, copy=False)[:3, :3]
        if repetition:
            sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
            if sy > _EPS:
                ax = math.atan2(M[i, j], M[i, k])
                ay = math.atan2(sy, M[i, i])
                az = math.atan2(M[j, i], -M[k, i])
            else:
                ax = math.atan2(-M[j, k], M[j, j])
                ay = math.atan2(sy, M[i, i])
                az = 0.0
        else:
            cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
            if cy > _EPS:
                ax = math.atan2(M[k, j], M[k, k])
                ay = math.atan2(-M[k, i], cy)
                az = math.atan2(M[j, i], M[i, i])
            else:
                ax = math.atan2(-M[j, k], M[j, j])
                ay = math.atan2(-M[k, i], cy)
                az = 0.0

        if parity:
            ax, ay, az = -ax, -ay, -az
        if frame:
            ax, az = az, ax
        return ax, ay, az

    @staticmethod
    def to_quaternion(transform, isprecise=False):
        """
        Return quaternion from the rotation part of an homogeneous transformation.

        Parameters
        ----------
        transform: array_like
            Rotation matrix. It can be (3x3) or (4x4)
        isprecise: bool
            If True, the input transform is assumed to be a precise rotation matrix and
            a faster algorithm is used.

        Returns
        -------
        q: array_like
            Quaternion in w, x, y z (real, then vector) format

        Notes
        -----
        Quaternions :math:`w + ix + jy + kz` are represented as :math:`[w, x, y, z]`.

        Examples
        --------
        >>> import numpy as np
        >>> import baldor as br
        >>> q = br.transform.to_quaternion(np.identity(4), isprecise=True)
        >>> np.allclose(q, [1, 0, 0, 0])
        True
        >>> q = br.transform.to_quaternion(np.diag([1, -1, -1, 1]))
        >>> np.allclose(q, [0, 1, 0, 0]) or np.allclose(q, [0, -1, 0, 0])
        True
        >>> T = br.axis_angle.to_transform((1, 2, 3), 0.123)
        >>> q = br.transform.to_quaternion(T, True)
        >>> np.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
        True
        """
        M = np.array(transform, dtype=np.float64, copy=False)[:4, :4]
        if isprecise:
            q = np.empty((4,))
            t = np.trace(M)
            if t > M[3, 3]:
                q[0] = t
                q[3] = M[1, 0] - M[0, 1]
                q[2] = M[0, 2] - M[2, 0]
                q[1] = M[2, 1] - M[1, 2]
            else:
                i, j, k = 0, 1, 2
                if M[1, 1] > M[0, 0]:
                    i, j, k = 1, 2, 0
                if M[2, 2] > M[i, i]:
                    i, j, k = 2, 0, 1
                t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
                q[i] = t
                q[j] = M[i, j] + M[j, i]
                q[k] = M[k, i] + M[i, k]
                q[3] = M[k, j] - M[j, k]
                q = q[[3, 0, 1, 2]]
            q *= 0.5 / math.sqrt(t * M[3, 3])
        else:
            m00 = M[0, 0]
            m01 = M[0, 1]
            m02 = M[0, 2]
            m10 = M[1, 0]
            m11 = M[1, 1]
            m12 = M[1, 2]
            m20 = M[2, 0]
            m21 = M[2, 1]
            m22 = M[2, 2]
            # symmetric matrix K
            K = np.array(
                [
                    [m00 - m11 - m22, 0.0, 0.0, 0.0],
                    [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                    [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                    [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
                ]
            )
            K /= 3.0
            # quaternion is eigenvector of K that corresponds to largest eigenvalue
            w, V = np.linalg.eigh(K)
            q = V[[3, 0, 1, 2], np.argmax(w)]
        if q[0] < 0.0:
            np.negative(q, q)
        return q


class br_vector:
    @staticmethod
    def unit(vector):
        """
        Return vector divided by Euclidean (L2) norm

        Parameters
        ----------
        vector: array_like
            The input vector

        Returns
        -------
        unit : array_like
            Vector divided by L2 norm

        Examples
        --------
        >>> import numpy as np
        >>> import baldor as br
        >>> v0 = np.random.random(3)
        >>> v1 = br.vector.unit(v0)
        >>> np.allclose(v1, v0 / np.linalg.norm(v0))
        True
        """
        v = np.asarray(vector).squeeze()
        return v / math.sqrt((v**2).sum())

    @staticmethod
    def norm(vector):
        """
        Return vector Euclidaan (L2) norm

        Parameters
        ----------
        vector: array_like
            The input vector

        Returns
        -------
        norm: float
            The computed norm

        Examples
        --------
        >>> import numpy as np
        >>> import baldor as br
        >>> v = np.random.random(3)
        >>> n = br.vector.norm(v)
        >>> numpy.allclose(n, np.linalg.norm(v))
        True
        """
        return math.sqrt((np.asarray(vector) ** 2).sum())

    @staticmethod
    def perpendicular(vector):
        """
        Find an arbitrary perpendicular vector

        Parameters
        ----------
        vector: array_like
            The input vector

        Returns
        -------
        result: array_like
            The perpendicular vector
        """
        if np.allclose(vector, np.zeros(3)):
            # vector is [0, 0, 0]
            raise ValueError("Input vector cannot be a zero vector")
        u = br_vector.unit(vector)
        if np.allclose(u[:2], np.zeros(2)):
            return Y_AXIS
        result = np.array([-u[1], u[0], 0], dtype=np.float64)
        return result

    @staticmethod
    def skew(vector):
        """
        Returns the 3x3 skew matrix of the input vector.

        The skew matrix is a square matrix `R` whose transpose is also its negative;
        that is, it satisfies the condition :math:`-R = R^T`.

        Parameters
        ----------
        vector: array_like
            The input array

        Returns
        -------
        R: array_like
            The resulting 3x3 skew matrix
        """
        skv = np.roll(np.roll(np.diag(np.asarray(vector).flatten()), 1, 1), -1, 0)
        return skv - skv.T

    @staticmethod
    def transform_between_vectors(vector_a, vector_b):
        """
        Compute the transformation that aligns two vectors

        Parameters
        ----------
        vector_a: array_like
            The initial vector
        vector_b: array_like
            The goal vector

        Returns
        -------
        transform: array_like
            The transformation between `vector_a` a `vector_b`
        """
        newaxis = br_vector.unit(vector_b)
        oldaxis = br_vector.unit(vector_a)
        # Limits the value of `c` to be within the range C{[-1, 1]}
        c = np.clip(np.dot(oldaxis, newaxis), -1.0, 1.0)
        angle = np.arccos(c)
        if np.isclose(c, -1.0) or np.allclose(newaxis, oldaxis):
            axis = br_vector.perpendicular(newaxis)
        else:
            axis = br_vector.unit(np.cross(oldaxis, newaxis))
        transform = br_axis_angle.to_transform(axis, angle)
        return transform
