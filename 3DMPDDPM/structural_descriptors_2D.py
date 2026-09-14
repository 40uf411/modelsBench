# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 10:42:27 2017

Python code for calculating various 2D structural indicators

@author: Laurent Lemmens <laurent.lemmens@sckcen.be>

"""

import numpy as np
from scipy import ndimage as ndi


def label_function(mt, target=1, Neighbors=4, ret=0):
    """
    label_function(mt, target=1, Neighbors=4, ret=0)
    Labels an array by assigning a value of 1 to all points that have a value of target, and a value of 0 to all other points.

    Parameters:
        mt: array to be labeled
        target: value to be labeled (default is 1)
        Neighbors: number of nearest points to use for labeling (4 or 8, default is 4)
        ret: if 0, only the labeled array is returned; if 1, the labeled array and the number of clusters are returned.

    Returns:
        if ret is 0, returns the labeled array.
        if ret is 1, returns a tuple containing the labeled array and the number of clusters.
    """
    mt1 = np.copy(mt)
    mt1[mt1 == target] = 255
    mt1[mt1 != 255] = 0
    mt1[mt1 == 255] = 1

    if Neighbors == 8:
        s = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        mt1, k = ndi.measurements.label(mt1, s)
    elif Neighbors == 4:
        mt1, k = ndi.measurements.label(mt1)
    if ret == 0:
        return mt1
    else:
        return mt1, k


def nboun(sgd, length):
    """
    Calculates the number of points on the diagonal for each leg distances
    sgd= size of the array
    Length= lag classes which should be investigated if integer(all lag distances till this vector are tested)
                                                     if array ( the array defines the tested lag classes)

    output  array with number of point combination for each leg distance                            
    """
    if np.size(length) == 1:
        if sgd[0] == sgd[1]:
            nboun1 = np.arange(length)
            npoins = (sgd[0]-nboun1)**2
            npoins = npoins.astype(float)
        elif sgd[0] > sgd[1]:
            p1 = np.arange(1, sgd[1]+1)
            p2 = 2*np.ones_like(p1)
            p2[-1] = sgd[0]-sgd[1]+1
            npoins = np.empty(length)
            for i in range(length):
                p_1 = np.copy(p1)
                p_1 = p_1-i

                p_1[p_1 < 0] = 0
                npoins[i] = np.dot(p_1, p2)
        elif sgd[0] < sgd[1]:
            p1 = np.arange(1, sgd[0]+1)
            p2 = 2*np.ones_like(p1)
            p2[-1] = sgd[1]-sgd[0]+1
            npoins = np.empty(length)
            for i in range(length):
                p_1 = np.copy(p1)
                p_1 = p_1-i
                p_1[p_1 < 0] = 0
                npoins[i] = np.dot(p_1, p2)
    else:
        if sgd[0] == sgd[1]:
            nboun1 = np.copy(length)
            npoins = (sgd[0]-nboun1)**2
            npoins = npoins.astype(int)
        elif sgd[0] > sgd[1]:
            p1 = np.arange(1, sgd[1]+1)
            p2 = 2*np.ones_like(p1)
            p2[-1] = sgd[0]-sgd[1]+1
            npoins = np.empty(len(length))
            for i in range(len(length)):
                p_1 = np.copy(p1)
                p_1 = p_1-length[i]
                p_1[p_1 < 0] = 0
                npoins[i] = np.dot(p_1, p2)
        elif sgd[0] < sgd[1]:
            p1 = np.arange(1, sgd[0]+1)
            p2 = 2*np.ones_like(p1)
            p2[-1] = sgd[1]-sgd[0]+1
            npoins = np.empty(len(length))
            for i in range(len(length)):
                p_1 = np.copy(p1)
                p_1 = p_1-length[i]
                p_1[p_1 < 0] = 0
                npoins[i] = np.dot(p_1, p2)

    return npoins


def resh_diag1(mt, length, row, col, posin, lagval=99):
    if np.size(length) == 1:
        mt_new = mt[row, col]
        s = lagval*np.ones(length, dtype=int)
        posin = np.tile(posin, len(s))
        posin = posin.astype(int)
        s = lagval*np.ones_like(posin, dtype=int)

        mt_new = np.insert(mt_new, posin, s)
    else:
        mt_new = mt[row, col]
        s = lagval*np.ones(length[-1])
        posin = np.tile(posin, len(s))
        s = lagval*np.ones_like(posin, dtype=int)
        mt_new = np.insert(mt_new, posin, s)
    return mt_new


def resh_diag2(mt1):
    x, y = np.shape(mt1)
    s = 0
    row = np.zeros(np.size(mt1), dtype=np.int32)
    col = np.zeros(np.size(mt1), dtype=np.int32)

    if x == y:
        posin = np.cumsum(np.append(np.arange(1, x+1), np.arange(x-1, 1, -1)))
        for i in range(1, x+1):
            h = (s+i)
            row[s:h] = np.arange(x-i, x)
            col[s:h] = np.arange(i)
            s = h

        for i in range(x-1, 0, -1):
            h = np.copy(s+i)
            row[s:h] = np.arange(i)
            col[s:h] = np.arange(x-i, x)
            s = h

    if x > y:
        k = 0
        times = x-y
        pos1 = np.arange(1, y+1)
        pos2 = np.append(y*np.ones(times), np.arange(y-1, 1, -1))
        posin = np.cumsum(np.append(pos1, pos2))
        for i in range(1, x+1):

            if i <= y:
                h = (s+i)
                row[s:h] = np.arange(x-i, x)
                col[s:h] = np.arange(i)
            else:
                h = s+y
                k += 1
                row[s:h] = np.arange(x-i, x-k)
                col[s:h] = np.arange(y)
            s = h
        for i in range(y-1, 0, -1):
            h = (s+i)
            row[s:h] = np.arange(i)
            col[s:h] = np.arange(y-i, y)
            s = h

    if x < y:
        times = y-x
        pos1 = np.arange(1, x+1)
        pos2 = np.append(x*np.ones(times), np.arange(x-1, 1, -1))
        posin = np.cumsum(np.append(pos1, pos2))
        k = 0
        for i in range(1, y+1):

            if i <= x:
                h = (s+i)
                row[s:h] = np.arange(x-i, x)
                col[s:h] = np.arange(i)
            else:
                h = (s+x)
                k += 1
                row[s:h] = np.arange(i-k)
                col[s:h] = np.arange(k, i)
            s = h
        for i in range(x-1, 0, -1):
            h = s+i
            row[s:h] = np.arange(i)
            col[s:h] = np.arange(y-i, y)
            s = h
    return row, col, posin


def same_facies(mt1, target=1, length=100, orientation='h', boundary='c', row=0, col=0, sgd=[], npoins=[], posin=0):
    """
    mt1= array which is investigated
    target= value which is tested for is occurrence if integer( than the label of phase which is tested)
                                                    if array (target should be the sum of the values which should be analyzed)
    Length= lag classes which should be investigated if integer(all lag distances till this vector are tested)
                                                     if array ( the array defines the tested lag classes)
    orientation ('h'=horizontal,'v'=vertical,'d'=diagonal) 
    boundary ('c'=continious boundary condition (2D array is flattend in to a vector),'u' uncontinious boundary condition (each line is treated independently)
    the rest  of the input are unnecessary information which would only speed up zour calculations for simulated anneling 
    output  if target== int() 1d array which gives the propability for each leg distances, leg 0 is Just the propabiltiy to find the specific value
            else each row is specific for one target value.
    """
    mt = np.copy(mt1)
    if np.size(length) == 1:
        if boundary == 'c':

            if orientation == 'h':
                mt = np.ravel(mt)
            elif orientation == 'v':
                mt = np.reshape(mt, -1, order='F')
            elif orientation == 'd':
                if np.size(row) == 1:
                    row, col, posin = resh_diag2(mt)

                mt = mt[row, col]

            x = np.size(mt)
            x = float(x)

            if np.size(target) == 1:
                s_f = np.zeros(length, dtype=float)
                s_f[0] = np.count_nonzero(mt == target)/x
                for i in range(1, length):
                    s_f[i] = np.count_nonzero(mt[:-i]+mt[i:] == 2*target)/(x-i)
            else:
                s_f = s_f = np.zeros((length, len(target)), dtype=float)
                for l in range(0, len(target)):
                    s_f[0, l] = np.count_nonzero(mt == target[l]/2)/x
                for i in range(1, length):
                    k = mt[:-i]+mt[i:]
                    for J in range(0, len(target)):

                        s_f[i, J] = np.count_nonzero(k == target[J])/(x-i)

        elif boundary == 'u':
            x = np.size(mt)

            if orientation == 'h':
                s = len(mt)

                x = float(x)
                if np.size(target) == 1:
                    s_f = np.zeros(length, dtype=float)
                    s_f[0] = np.count_nonzero(mt == target)/x

                    for i in range(1, length):
                        s_f[i] = np.count_nonzero(
                            mt[:, :-i]+mt[:, i:] == 2*target)/(x-i*s)
                else:
                    s_f = np.zeros((length, len(target)), dtype=float)
                    for J in range(0, len(target)):
                        s_f[0, J] = np.count_nonzero(mt == target[J]/2)/x
                    for i in range(1, length):
                        k = mt[:, :-i]+mt[:, i:]
                        for J in range(0, len(target)):

                            s_f[i, J] = np.count_nonzero(
                                k == target[J])/(x-i*s)
            elif orientation == 'v':
                s = len(mt[0])
                x = float(x)
                if np.size(target) == 1:
                    s_f = np.zeros(length, dtype=float)
                    s_f[0] = np.count_nonzero(mt == target)/x
                    for i in range(1, length):
                        s_f[i] = np.count_nonzero(
                            mt[:-i, :]+mt[i:, :] == 2*target)/(x-i*s)
                else:
                    s_f = np.zeros((length, len(target)), dtype=float)
                    for J in range(0, len(target)):
                        s_f[0, J] = np.count_nonzero(mt == target[J]/2)/x

                    for i in range(1, length):
                        k = mt[:-i, :]+mt[i:, :]
                        for J in range(0, len(target)):

                            s_f[i, J] = np.count_nonzero(
                                k == target[J])/(x-i*s)
            elif orientation == 'd':
                x = float(x)
                if np.size(row) == 1:
                    row, col, posin = resh_diag2(mt)
                mt = resh_diag1(mt, length, row, col, posin)
                if len(npoins) == 0:
                    if len(sgd) == 0:
                        sgd = np.array(np.shape(mt1))
                    npoins = nboun(sgd, length)
                if np.size(target) == 1:
                    s_f = np.zeros(length, dtype=float)
                    s_f[0] = np.count_nonzero(mt == target)/x
                    for i in range(1, length):

                        s_f[i] = np.count_nonzero(
                            mt[:-i]+mt[i:] == 2*target)/(npoins[i])

                else:
                    s_f = np.zeros((length, len(target)), dtype=float)
                    for J in range(0, len(target)):
                        s_f[0, J] = np.count_nonzero(mt == target[J]/2)/x

                    for i in range(1, length):
                        k = mt[:-i]+mt[i:]
                        for J in range(0, len(target)):

                            s_f[i, J] = np.count_nonzero(
                                k == target[J])/(npoins[i])

    else:
        if boundary == 'c':

            if orientation == 'h':
                mt = np.reshape(mt, -1)
            elif orientation == 'v':
                mt = np.reshape(mt, -1, order='F')
            elif orientation == 'd':
                if np.size(row) == 1:
                    row, col, posin = resh_diag2(mt)
                    mt = mt[row, col]
                else:
                    mt = mt[row, col]

            x = np.size(mt)
            x = float(x)
            if np.size(target) == 1:
                s_f = np.zeros(length[-1]+1, dtype=float)

                if length[0] == 0:

                    s_f[0] = np.count_nonzero(mt == target)/x
                    for i in range(1, len(length)):
                        s_f[i] = np.count_nonzero(
                            mt[:-length[i]]+mt[length[i]:] == 2*target)/(x-length[i])
                else:
                    for i in range(len(length)):
                        s_f[i] = np.count_nonzero(
                            mt[:-length[i]]+mt[length[i]:] == 2*target)/(x-length[i])
            else:
                s_f = np.zeros((len(length), len(target)), dtype=float)
                if length[0] == 0:
                    for l in range(0, len(target)):
                        s_f[0, l] = np.count_nonzero(mt == target[l])/x
                    for i in range(1, len(length)):
                        k = mt[:-length[i]]+mt[length[i]:]
                        for J in range(0, len(target)):

                            s_f[i, J] = np.count_nonzero(
                                k == target[J])/(x-length[i])
                else:
                    for i in range(len(length)):
                        k = mt[:-i]+mt[i:]
                        for J in range(0, len(target)):

                            s_f[i, J] = np.count_nonzero(
                                k == target[J])/(x-length[i])

        elif boundary == 'u':
            x = np.size(mt)
            x = float(x)
            if orientation == 'h':
                s = len(mt[0])
                if np.size(target) == 1:
                    s_f = np.zeros(len(length), dtype=float)
                    if length[0] == 0:
                        s_f[0] = np.count_nonzero(mt == target)/x
                        for i in range(1, len(length)):
                            s_f[i] = np.count_nonzero(
                                mt[:, :-length[i]]+mt[:, length[i]:] == 2*target)/(x-length[i] * s)
                    else:
                        for i in range(len(length)):
                            s_f[i] = np.count_nonzero(
                                mt[:, :-length[i]]+mt[:, length[i]:] == 2*target)/(x-length[i] * s)
                else:
                    s_f = np.zeros((len(length), len(target)), dtype=float)
                    if length[0] == 0:
                        for l in range(0, len(target)):
                            s_f[0, l] = np.count_nonzero(mt == target[l])/x
                        for i in range(1, len(length)):
                            k = mt[:, :-length[i]]+mt[:, length[i]:]
                            for l in range(0, len(target)):

                                s_f[i, l] = np.count_nonzero(
                                    k == target[l])/(x-length[i] * s)
                    else:
                        for i in range(len(length)):
                            k = mt[:, :-length[i]]+mt[:, length[i]:]
                            for l in range(0, len(target)):

                                s_f[i, l] = np.count_nonzero(
                                    k == target[l])/(x-length[i]*s)

            elif orientation == 'v':
                s = len(mt)

                if np.size(target) == 1:
                    s_f = np.zeros(len(length), dtype=float)
                    if length[0] == 0:
                        s_f[0] = np.count_nonzero(mt == target)/x
                        for i in range(1, len(length)):
                            s_f[i] = np.count_nonzero(
                                mt[:-length[i], :]+mt[length[i]:, :] == 2*target)/(x-length[i] * s)
                    else:
                        for i in range(len(length)):
                            s_f[i] = np.count_nonzero(
                                mt[:-length[i], :]+mt[length[i]:, :] == 2*target)/(x-length[i] * s)
                else:
                    s_f = np.zeros((len(length), len(target)), dtype=float)
                    if length[0] == 0:
                        for l in range(0, len(target)):
                            s_f[0, l] = np.count_nonzero(mt == target[l]/2)/x
                        for i in range(1, len(length)):
                            k = mt[:-length[i], :]+mt[length[i]:, :]
                            for l in range(0, len(target)):

                                s_f[i, l] = np.count_nonzero(
                                    k == target[l])/(x-length[i] * s)
                    else:
                        for i in range(len(length)):
                            k = mt[:-length[i], :]+mt[length[i]:, :]
                            for l in range(0, len(target)):

                                s_f[i, l] = np.count_nonzero(
                                    k == target[l])/(x-length[i] * s)
            elif orientation == 'd':
                if np.size(row) == 1:
                    row, col, posin = resh_diag2(mt)
                mt = resh_diag1(mt, length, row, col, posin)
                if len(npoins) == 0:
                    if len(sgd) == 0:
                        sgd = np.array(np.shape(mt1))
                    npoins = nboun(sgd, length)
                if np.size(target) == 1:
                    s_f = np.zeros(len(length), dtype=float)
                    if length[0] == 0:
                        s_f[0] = np.count_nonzero(mt == target)/x
                        for i in range(1, len(length)):
                            s_f[i] = np.count_nonzero(
                                mt[:-length[i]]+mt[length[i]:] == 2*target)/(npoins[i])
                    else:
                        for i in range(len(length)):
                            s_f[i] = np.count_nonzero(
                                mt[:-length[i]]+mt[length[i]:] == 2*target)/(npoins[i])
                else:
                    s_f = np.zeros((len(length), len(target)), dtype=float)
                    if length[0] == 0:
                        for l in range(0, len(target)):
                            s_f[0, l] = np.count_nonzero(mt == target[l]/2)/x
                        for i in range(1, len(length)):
                            k = mt[:-length[i]]+mt[length[i]:]
                            for l in range(0, len(target)):

                                s_f[i, l] = np.count_nonzero(
                                    k == target[l])/(npoins[i])
                    else:
                        for i in range(len(length)):
                            k = mt[:-length[i]]+mt[length[i]:]
                            for l in range(0, len(target)):

                                s_f[i, l] = np.count_nonzero(
                                    k == target[l])/(npoins[i])
    return s_f


def lineal_function(mt1, target=1, length=100, orientation='h', boundary='c', row=0, col=0, sgd=[], npoins=[], posin=0):
    """
    Lineal path function
    mt1= array which is investigated
    target= value which is tested for is occurrence (only one phase at a time is possible)
    Length= lag classes which should be investigated if integer(all lag distances till this vector are tested)
                                                     if array ( the array defines the tested lag classes)
    orientation ('h'=horizontal,'v'=vertical,'d'=diagonal) 
    boundary ('c'=continious boundary condition (2D array is flattend in to a vector),'u' uncontinious boundary condition (each line is treated independently)
    the rest  of the input are unnecessary information which would only speed up your calculations for simulated anneling, if you calculate the descriptor along its diagonal
                if you do not specify those values explicitly beforehand  those values are calculated if needed
    output  if target== int() 1d array which gives the propability for each leg distances, leg 0 is just the propabiltiy to find the specific value.
    """
    mt = np.copy(mt1)
    if np.size(length) == 1:
        if boundary == 'c':
            if orientation == 'h':
                mt = np.reshape(mt, -1)
            elif orientation == 'v':
                mt = np.reshape(mt, -1, order='F')
            elif orientation == 'd':
                if np.size(row) == 1:
                    row, col, posin = resh_diag2(mt)
                mt = mt[row, col]
            l_ine = np.zeros(length)
            x = np.size(mt)
            x = float(x)
            t = np.nonzero(mt == target)[0]
            l_ine[0] = len(t)/x
            for i in range(1, length):
                l_ine[i] = np.count_nonzero(t[i:]-t[:-i] == i)/(x-i)
                if l_ine[i] == 0:
                    break
        elif boundary == 'u':
            x = np.size(mt)
            x = float(x)
            if orientation == 'h':
                j = (len(mt), length)
                m = 0.1*np.ones(j)
                mt = np.hstack((mt, m))
                mt = np.reshape(mt, -1)
                l_ine = np.zeros(length)
                t = np.nonzero(mt == target)[0]
                l_ine[0] = len(t)/x
                for i in range(1, length):
                    l_ine[i] = np.count_nonzero(t[i:]-t[:-i] == i)/(x-i*j[0])
                    if l_ine[i] == 0:
                        break
            elif orientation == 'v':
                j = (length, len(mt[0]))
                m = 0.1*np.ones(j)
                mt = np.vstack((mt, m))
                mt = np.reshape(mt, -1, order='F')
                l_ine = np.zeros(length)
                t = np.nonzero(mt == target)[0]
                l_ine[0] = len(t)/x
                for i in range(1, length):
                    l_ine[i] = np.count_nonzero(t[i:]-t[:-i] == i)/(x-i*j[1])
                    if l_ine[i] == 0:
                        break
            elif orientation == 'd':
                if len(npoins) == 0:
                    if len(sgd) == 0:
                        sgd = np.array(np.shape(mt))
                    npoins = nboun(sgd, length)
                if np.size(row) == 1:
                    row, col, posin = resh_diag2(mt)

                mt = resh_diag1(mt, length, row, col, posin)

                l_ine = np.zeros(length)
                t = np.nonzero(mt == target)[0]
                l_ine[0] = len(t)/x
                for i in range(1, length):
                    l_ine[i] = np.count_nonzero(t[i:]-t[:-i] == i)/(npoins[i])
                    if l_ine[i] == 0:
                        break
    else:
        if boundary == 'c':
            if orientation == 'h':
                mt = np.reshape(mt, -1)
            elif orientation == 'v':
                mt = np.reshape(mt, -1, order='F')
            elif orientation == 'd':
                if np.size(row) == 1:
                    row, col, posin = resh_diag2(mt)
                mt = mt[row, col]
            l_ine = np.zeros(len(length))
            x = np.size(mt)
            x = float(x)
            t = np.nonzero(mt == target)[0]
            if length[0] == 0:
                l_ine[0] = len(t)/x
                for i in range(1, len(length)):
                    l_ine[i] = np.count_nonzero(
                        t[length[i]:]-t[:-length[i]] == length[i])/(x-length[i])
                    if l_ine[i] == 0:
                        break
            else:
                for i in range(len, length):
                    l_ine[i] = np.count_nonzero(
                        t[length[i]:]-t[:-length[i]] == length[i])/(x-length[i])
                    if l_ine[i] == 0:
                        break
        elif boundary == 'u':
            x = np.size(mt)
            x = float(x)
            if orientation == 'h':
                j = (len(mt), length[-1])
                m = 0.1*np.ones(j)
                mt = np.hstack((mt, m))
                mt = np.reshape(mt, -1)
                t = np.nonzero(mt == target)[0]
                l_ine = np.zeros(len(length))
                if length[0] == 0:
                    l_ine[0] = len(t)/x
                    for i in range(1, len(length)):
                        l_ine[i] = np.count_nonzero(
                            t[length[i]:]-t[:-length[i]] == length[i])/(x-length[i]*j[0])
                        if l_ine[i] == 0:
                            break
                else:
                    for i in range(len(length)):
                        l_ine[i] = np.count_nonzero(
                            t[length[i]:]-t[:-length[i]] == length[i])/(x-length[i]*j[0])
                        if l_ine[i] == 0:
                            break
            elif orientation == 'v':
                j = (length[-1], len(mt[0]))
                m = 0.1*np.ones(j)
                mt = np.vstack((mt, m))
                mt = np.reshape(mt, -1, order='F')
                t = np.nonzero(mt == target)[0]
                l_ine = np.zeros(len(length))
                if length[0] == 0:
                    l_ine[0] = len(t)/x
                    for i in range(1, len(length)):
                        l_ine[i] = np.count_nonzero(
                            t[length[i]:]-t[:-length[i]] == length[i])/(x-length[i]*j[1])
                        if l_ine[i] == 0:
                            break
                else:
                    for i in range(len(length)):
                        l_ine[i] = np.count_nonzero(
                            t[length[i]:]-t[:-length[i]] == length[i])/(x-length[i]*j[1])
                        if l_ine[i] == 0:
                            break
            elif orientation == 'd':
                if np.size(row) == 1:
                    row, col, posin = resh_diag2(mt)
                mt = resh_diag1(mt, length, row, col, posin)
                t = np.nonzero(mt == target)[0]
                l_ine = np.zeros(len(length))
                if length[0] == 0:
                    l_ine[0] = len(t)/x
                    for i in range(1, len(length)):
                        l_ine[i] = np.count_nonzero(
                            t[length[i]:]-t[:-length[i]] == length[i])/(npoins[i])
                        if l_ine[i] == 0:
                            break
                else:
                    for i in range(len(length)):
                        l_ine[i] = np.count_nonzero(
                            t[length[i]:]-t[:-length[i]] == length[i])/(npoins[i])
                        if l_ine[i] == 0:
                            break
    return l_ine


def cluster_function(mt1, target=1, length=100, orientation='h', boundary='c', row=0, col=0, sgd=[], npoins=[], Neighbors=4, label=[], posin=0):
    """
    Two point cluster function
    mt1= array which is investigated
    target= value which is tested for is occurrence (only one phase at a time is possible)
    Length= lag classes which should be investigated if integer(all lag distances till this vector are tested)
                                                     if array ( the array defines the tested lag classes)
    orientation ('h'=horizontal,'v'=vertical,'d'=diagonal) 
    boundary ('c'=continious boundary condition (2D array is flattend in to a vector),'u' uncontinious boundary condition (each line is treated independently)
    the rest  of the input are unnecessary information which would only speed up your calculations for simulated anneling,
                if you do not specify those values explicitly beforehand  those values are calculated if needed
    output  1d array which gives the propability for each leg distances, leg 0 is just the propabiltiy to find the specific value
    """
    if len(label) == 0:
        mt = label_function(mt1, target, Neighbors)
    else:
        mt = np.copy(label)

    if np.size(length) == 1:
        if boundary == 'c':

            if orientation == 'h':
                mt = np.ravel(mt)
            elif orientation == 'v':
                mt = np.reshape(mt, -1, order='F')
            elif orientation == 'd':
                if np.size(row) == 1:
                    row, col, posin = resh_diag2(mt)
                mt = mt[row, col]
            x = np.size(mt)
            x = float(x)
            c_f = np.zeros(length, dtype=float)
            c_f[0] = np.count_nonzero(mt != 0)/x
            mt0 = mt != 0
            for i in range(1, length):
                c_f[i] = np.count_nonzero((mt[:-i] == mt[i:])*mt0[:-i])/(x-i)
        elif boundary == 'u':
            x = np.size(mt)
            x = float(x)

            if orientation == 'h':
                j = (len(mt), length)
                m = np.zeros(j)
                mt = np.hstack((mt, m))
                mt = np.reshape(mt, -1)
                c_f = np.zeros(length, dtype=float)
                x = float(x)
                c_f[0] = np.count_nonzero(mt != 0)/x
                mt0 = mt != 0
                for i in range(1, length):
                    c_f[i] = np.count_nonzero(
                        (mt[:-i] == mt[i:]) * mt0[:-i])/(x-i*j[0])
            elif orientation == 'v':
                j = (length, len(mt[0]))
                m = np.zeros(j)
                mt = np.vstack((mt, m))
                mt = np.reshape(mt, -1, order='F')
                c_f = np.zeros(length, dtype=float)
                x = float(x)
                c_f[0] = np.count_nonzero(mt)/x
                mt0 = mt != 0
                for i in range(1, length):
                    c_f[i] = np.count_nonzero(
                        (mt[:-i] == mt[i:]) * mt0[:-i])/(x-i*j[1])
            elif orientation == 'd':
                if np.size(row) == 1:
                    row, col, posin = resh_diag2(mt)
                mt = resh_diag1(mt, length, row, col, posin, 0)
                if len(npoins) == 0:
                    if len(sgd) == 0:
                        sgd = np.array(np.shape(mt1))

                    npoins = nboun(sgd, length)
                c_f = np.zeros(length, dtype=float)
                x = float(x)
                c_f[0] = np.count_nonzero(mt != 0)/x
                mt0 = mt != 0
                for i in range(1, length):

                    c_f[i] = np.count_nonzero(
                        (mt[:-i] == mt[i:]) * mt0[:-i])/(npoins[i])
    else:
        if boundary == 'c':

            if orientation == 'h':
                mt = np.reshape(mt, -1)
            elif orientation == "v":
                mt = np.reshape(mt, -1, order='F')
            elif orientation == "d":
                if np.size(row) == 1:
                    row, col, posin = resh_diag2(mt)
                mt = mt[row, col]

            x = np.size(mt)
            x = float(x)
            c_f = np.zeros(len(length))
            mt0 = mt != 0
            if length[0] == 0:
                c_f[0] = np.count_nonzero(mt != 0)/x
                for i in range(1, len(length)):
                    c_f[i] = np.count_nonzero(
                        (mt[:-length[i]] == mt[length[i]:])*mt0[:-length[i]])/(x-length[i])
            else:
                for i in range(len(length)):
                    c_f[i] = np.count_nonzero(
                        (mt[:-length[i]] == mt[length[i]:])*mt0[:-length[i]])/(x-length[i])
        else:
            x = np.size(mt)
            x = float(x)
            if orientation == 'h':

                j = (len(mt), length[-1])
                m = np.zeros(j)
                mt = np.hstack((mt, m))
                mt = np.reshape(mt, -1)
                mt0 = mt != 0
                c_f = np.zeros(len(length), dtype=float)
                if length[0] == 0:
                    c_f[0] = np.count_nonzero(mt != 0)/x
                    for i in range(1, len(length)):
                        c_f[i] = np.count_nonzero(
                            (mt[:-length[i]] == mt[length[i]:])*mt0[:-length[i]])/(x-length[i]*j[0])
                else:
                    for i in range(len(length)):
                        c_f[i] = np.count_nonzero(
                            (mt[:-length[i]] == mt[length[i]:])*mt0[:-length[i]])/(x-length[i]*j[0])
            elif orientation == 'v':
                j = (length[-1], len(mt[0]))
                m = np.zeros(j)
                mt = np.vstack((mt, m))
                mt = np.reshape(mt, -1, order='F')
                mt0 = mt != 0
                c_f = np.zeros(len(length), dtype=float)
                if length[0] == 0:
                    c_f[0] = np.count_nonzero(mt != 0)/x
                    for i in range(1, len(length)):
                        c_f[i] = np.count_nonzero(
                            (mt[:-length[i]] == mt[length[i]:])*mt0[:-length[i]])/(x-length[i]*j[1])
                else:
                    for i in range(len(length)):
                        c_f[i] = np.count_nonzero(
                            (mt[:-length[i]] == mt[length[i]:])*mt0[:-length[i]])/(x-length[i]*j[0])
            elif orientation == 'd':
                mt0 = mt != 0
                if np.size(row) == 1:
                    row, col, posin = resh_diag2(mt)
                mt = resh_diag1(mt, length, row, col, posin, 0)
                if len(npoins) == 0:
                    if len(sgd) == 0:
                        sgd = np.array(np.shape(mt1))

                    npoins = nboun(sgd, length)
                c_f = np.zeros(len(length), dtype=float)
                mt0 = mt != 0
                if length[0] == 0:
                    c_f[0] = np.count_nonzero(mt != 0)/x
                    for i in range(1, len(length)):
                        c_f[i] = np.count_nonzero(
                            (mt[:-length[i]] == mt[length[i]:]) * mt0[:-length[i]])/(npoins[i])
                else:
                    for i in range(len(length)):
                        c_f[i] = np.count_nonzero(
                            (mt[:-length[i]] == mt[length[i]:]) * mt0[:-length[i]])/(npoins[length[i]])

    return c_f


def pore_size_function(mt1, target=1, bins=np.arange(0.1, 1, 0.1), Neighbors=4, lab=[], k=0):
    """
    To calculate the pore size distribution logarithmic mean of  interval classes is used
    mt1= array which is investigated
    target= value which is tested for is occurrence (only one phase at a time is possible)
    bins= defines the classes  which are used for mean calculation
   Neighbours= if 4 the 4 nearest points are used for labeling
                if 8 the 8 nearest points are used for labeling
    the rest  of the input are unnecessary information which would only speed up your calculations for simulated anneling,
                if you do not specify those values explicitly beforehand  those values are calculated if needed (k is the number of clusters)
    output  1d array which gives the  logaritmic mean pore size for each class,
            1d array which gave the number of points in each cluster
    """
    if len(lab) == 0 or (k) == 0:
        lab, k = label_function(mt1, target, Neighbors, 1)
#    freq=np.bincount(np.reshape(mt,-1))[1:]
    binc = np.bincount(np.reshape(lab, -1))[1:]
    binc_sort = np.sort(binc)
    binc_sort10 = np.log10(binc_sort)
    bins1 = bins*k
    bins1 = bins1.astype(int)
    pore_mean = np.zeros(len(bins1)-1)

    for I in range(len(pore_mean)):
        pore_mean[I] = np.mean(binc_sort10[bins1[I]:bins1[I+1]])
#    % por=np.log10(np.percentile(binc,bins))
    return pore_mean, binc


def pore_perimeter(mt1, target=0, bins=np.arange(0.1, 1, 0.10), Neighbors=4, lab=[], k=0, bord=[]):
    """
    To calculate the perimeter distribution  the logarithmic mean of  interval classes is used
    mt1= array which is investigated
    target= value which is tested for is occurance (only one phase at a time is possible)
    bins= defines the classes  which are used for mean calculation
    Neighbours= if 4 the 4 nearest points are used for labeling
                if 8 the 8 nearest points are used for labeling
    the rest  of the input are unnecessary information which would only speed up your calculations for simulated anneling,
                if you do not specify those values explicitly beforehand  those values are calculated if needed (k is the number of clusters)
    output  1d array which gives the mean logaritmic perimeter  for each class,
            1d array which gave perimeter of each cluster
    """
    if len(lab) == 0 or k == 0:
        lab, k = label_function(mt1, target, Neighbors, 1)
    if len(bord) == 0:
        bord = borderopt2(mt1)

#    mt_2=ndi.labeled_comprehension(mt_2, mt, val, np.sum, float, 0)
    val = range(1, k+1)
    per = ndi.measurements.sum(bord, lab, val)
    per_sort = np.sort(per)
    per_sort10 = np.log10(per_sort)
    bins1 = bins*k
    bins1 = bins1.astype(int)
    pore_meanp = np.zeros(len(bins1)-1)
    for I in range(len(pore_meanp)):
        pore_meanp[I] = np.mean(per_sort10[bins1[I]:bins1[I+1]])
#        if np.isinf(pore_meanp[I])==1:
#                pdb.set_trace()

    return pore_meanp, per


def borderopt2(sg):
    """
    Calculates the number of various neighbours assuming a 4 point no
    sg= array which is investigated
    target= value which is tested for is occurrence (only one phase at a time is possible)
    bins= defines the classes  which are used for mean calculation
    Neighbours= if 4 the 4 nearest points are used for labeling
                if 8 the 8 nearest points are used for labeling
    the rest  of the input are unnecessary information which would only speed up your calculations for simulated anneling,
                if you do not specify those values explicitly beforehand  those values are calculated if needed (k is the number of clusters)
    output  1d array which gives the mean logarithmic perimeter  for each class,
            1d array which gave perimeter of each cluster
    """
    x, y = sg.shape
    mt_1 = np.zeros((x+2, y+2), dtype=float)
    mt_1[1:-1, 1:-1] = np.copy(sg)
    mt_1[0, 1:-1] = np.copy(sg[0, :])
    mt_1[-1, 1:-1] = np.copy(sg[-1, :])
    mt_1[1:-1, 0] = np.copy(sg[:, 0])
    mt_1[1:-1, -1] = np.copy(sg[:, -1])
    mt_2 = np.fabs(4-(mt_1[:-2, 1:-1] == mt_1[1:-1, 1:-1])-(mt_1[2:, 1:-1] == mt_1[1:-1, 1:-1])-(
        mt_1[1:-1, 2:] == mt_1[1:-1, 1:-1])-(mt_1[1:-1, :-2] == mt_1[1:-1, 1:-1]))
    return mt_2
