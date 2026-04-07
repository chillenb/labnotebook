import scipy
import numpy as np
from pyscf import lib
import time

def zherk_Lpq_to_eri(Lpq, eri=None):
    assert Lpq.flags.c_contiguous
    naux = Lpq.shape[0]
    nmo = Lpq.shape[1]
    if eri is None:
        eri = np.zeros((nmo*nmo, nmo*nmo), dtype=np.complex128)
    scipy.linalg.blas.zherk(
                    alpha=1.0,
                    a=Lpq.reshape(naux, nmo * nmo).T,
                    beta=0.0,
                    c=eri.T,
                    overwrite_c=True,
    )
    lib.hermi_triu(eri, inplace=True)
    return eri.reshape(nmo, nmo, nmo, nmo)

def zgemm_Lpq_to_eri(Lpq, eri=None):
    assert Lpq.flags.c_contiguous
    naux = Lpq.shape[0]
    nmo = Lpq.shape[1]
    if eri is None:
        eri = np.zeros((nmo*nmo, nmo*nmo), dtype=np.complex128)
    scipy.linalg.blas.zgemm(
                    alpha=1.0,
                    a=Lpq.reshape(naux, nmo * nmo).T,
                    b=Lpq.reshape(naux, nmo * nmo).T,
                    trans_a=0,
                    trans_b=2,
                    beta=0.0,
                    c=eri.T,
                    overwrite_c=True,
    )
    lib.hermi_triu(eri, inplace=True)
    return eri.reshape(nmo, nmo, nmo, nmo)

def test_zherk(nmo, naux):
    Lpq = np.random.rand(naux, nmo, nmo) + 1j * np.random.rand(naux, nmo, nmo)
    eri = zherk_Lpq_to_eri(Lpq)
    eri_zgemm = zgemm_Lpq_to_eri(Lpq)
    # this expression may differ from the actual ERI definition.
    # this script is for performance measurement.
    eri_ref = np.einsum("Lrs,Lpq->pqrs", Lpq, Lpq.conj())
    print(np.linalg.norm(eri.real-eri_ref.real))
    print(np.linalg.norm(eri.imag-eri_ref.imag))
    print(np.linalg.norm(eri_zgemm.real-eri_ref.real))
    print(np.linalg.norm(eri_zgemm.imag-eri_ref.imag))


def bench_zherk(nmo, naux):
    Lpq = np.random.rand(naux, nmo, nmo) + 1j * np.random.rand(naux, nmo, nmo)
    eri = np.empty((nmo*nmo, nmo*nmo), dtype=np.complex128)
    t0 = time.perf_counter()
    for _ in range(10):
        zherk_Lpq_to_eri(Lpq, eri)
    t1 = time.perf_counter()
    ms_per_run = int((t1-t0)/10*1000)
    print("zherk: (ms/run) ", ms_per_run)
    t0 = time.perf_counter()
    for _ in range(10):
        zgemm_Lpq_to_eri(Lpq, eri)
    t1 = time.perf_counter()
    ms_per_run = int((t1-t0)/10*1000)
    print("zgemm: (ms/run) ", ms_per_run)


if __name__ == '__main__':
    test_zherk(10, 50)
    nmo = 70
    naux = nmo * 5
    bench_zherk(nmo, naux)