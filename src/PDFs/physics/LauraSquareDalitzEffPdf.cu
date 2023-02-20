#include <goofit/PDFs/ParameterContainer.h>
#include <goofit/PDFs/physics/DalitzPlotHelpers.h>
#include <goofit/PDFs/physics/LauraSquareDalitzEffPdf.h>
#include <vector>
#include <math.h>

namespace GooFit {

__device__ auto thetaprime( fptype s12, fptype s13) -> fptype {  //s_ij, s_ik : i = 1, j = 2, k = 3
    fptype m1 = 0.13957061;
    fptype m2=0.13957061; 
    fptype m3=0.497614;
    fptype m123=1.86483;
    fptype m12 = sqrt(s12);
    fptype e1cm12 = ((m12*m12) - (m2*m2) + (m1*m1))/(2*m12); //centre of mass energy
    fptype e3cm12 = ((m123*m123) - (m12*m12) + (m3*m3))/(2*m12);
    fptype qi = sqrt((e1cm12*e1cm12) - (m1*m1));
    fptype qk = sqrt((e3cm12*e3cm12) - (m3*m3));
    fptype cos = (s13 - (m1*m1) - (m3*m3) - (2.0*e1cm12*e3cm12))/(2.0*qi*qk);
    if (cos > 1.0){cos = 1.0;} //force within physical limit. discontinuity handled by Jaccobian
    if (cos < -1.0){cos = -1.0;}
    return acos(cos)/M_PI;
    }

__device__ auto  mprime(fptype s12) -> fptype {
    /*
    calculate m' for square dalitz formalism. 
    :param s12: mass s12
    :returns: m'
    */
    fptype m12 = sqrt(s12);
    fptype delta_m12 = 1.367216 - 0.27914122; //hardcoded for Kpi
    fptype a = (m12 - 0.27914122)/delta_m12;
    return (1/M_PI)*acos((2*a) - 1);
    }
   
__device__ auto device_LauraSquareDalitzEff(fptype *evt, ParameterContainer &pc) -> fptype {
    // Define observables
    int idx = pc.getObservable(0);
    int idy = pc.getObservable(1);

    // don't use RO_CACHE as this is used as efficiency for Amp3Body
    fptype x = evt[idx];
    fptype y = evt[idy];

    // Define coefficients
    fptype c0 = pc.getParameter(0);
    fptype c1 = pc.getParameter(1);
    fptype c2 = pc.getParameter(2);
    fptype c3 = pc.getParameter(3);
    fptype c4 = pc.getParameter(4);
    fptype c5 = pc.getParameter(5);
    fptype c6 = pc.getParameter(6);
    fptype c7 = pc.getParameter(7);
    fptype c8 = pc.getParameter(8);
    fptype c9 = pc.getParameter(9);
    fptype c10 = pc.getParameter(10);
    fptype c11 = pc.getParameter(11);
    fptype c12 = pc.getParameter(12);
    fptype c13 = pc.getParameter(13);


    fptype mD   = 1.86483;
    fptype mKS0 = 0.497611;
    fptype mh1  = 0.13957;
    fptype mh2  = 0.13957;

    pc.incrementIndex(1, 8, 0, 2, 1);

    // Check phase space
    if(!inDalitz(x, y, mD, mKS0, mh1, mh2))
        return 0;

    // Call helper functions
    fptype tp = thetaprime(x, y);
    if(tp > 1. || tp < 0.)
        return 0;
    
    fptype mp = mprime(x);
    if(mp > 1. || mp < 0.)
        return 0;

    fptype ret = c0*mp + c1*tp +c2*mp*tp + c3*mp*mp + c4*tp*tp + c5*mp*tp*tp + c6*tp*mp*mp 
               + c7*tp*tp*tp + c8*mp*mp*mp + c9*mp*mp*tp*tp + c10*mp*tp*tp*tp 
               + c11*tp*tp*tp*tp + c12*mp*mp*mp*mp +c13*mp*mp*mp*tp;

    return ret;
}

__device__ device_function_ptr ptr_to_LauraSquareDalitzEff = device_LauraSquareDalitzEff;

__host__ LauraSquareDalitzEffPdf::LauraSquareDalitzEffPdf(std::string n,
                                                Observable m12,
                                                Observable m13,
                                                Variable c0,
                                                Variable c1,
                                                Variable c2,
                                                Variable c3,
                                                Variable c4,
                                                Variable c5,
                                                Variable c6,
                                                Variable c7,
                                                Variable c8,
                                                Variable c9,
                                                Variable c10,
                                                Variable c11,
                                                Variable c12,
                                                Variable c13)

    : GooPdf("LauraSquareDalitzEffPdf", n, m12, m13, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13) {
    registerFunction("ptr_to_LauraSquareDalitzEff", ptr_to_LauraSquareDalitzEff);

    initialize();
}

// __host__ fptype LauraSquareDalitzEffPdf::normalize() { return 1; }

} // namespace GooFit
