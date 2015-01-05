#include"algorithm/LinearAlgebra.h"
#include"data/functor/FunctorF.h"
#include"algorithm/ForEachFunctor.h"
namespace pop{




Mat2F64 LinearAlgebra::random(int size_i,int size_j, Distribution  proba){
    Mat2F64 m(size_i,size_j);
    Mat2F64::iterator __first = m.begin();
    Mat2F64::iterator __last = m.end();
    for (; __first != __last; ++__first)
        *__first = proba.randomVariable();
    return m;
}

Mat2F64 LinearAlgebra::inverseGaussianElimination(const Mat2F64 &m){
    Mat2F64 M(m.sizeI(),m.sizeJ()*2);
    for(unsigned int i =0;i<m.sizeI();i++)
        for(unsigned int j =0;j<m.sizeJ();j++)
            M(i,j)=m(i,j);
    for(unsigned int i=0;i<m.sizeI();i++){
        M(i,m.sizeJ()+i)=1;
    }
    M= LinearAlgebra::solvingLinearSystemGaussianElimination(M);
    Mat2F64 Mout(m.sizeI(),m.sizeJ());
    for(unsigned int i=0;i<m.sizeI();i++){
        for(unsigned int j=0;j<m.sizeJ();j++){
            Mout(i,j)=M(i,m.sizeJ()+j);
        }
    }
    return Mout;
}

Mat2F64 LinearAlgebra::orthogonalGramSchmidt(const Mat2F64& m)throw(pexception)
{
    if(m.sizeI()!=m.sizeI())
        throw(pexception("In linearAlgebra::orthogonalGramSchmidt, Mat2F64 must be square"));
    Vec<VecF64> u(m.sizeI(),VecF64(m.sizeI()));
    for(unsigned int k=0;k<m.sizeI();k++){
        VecF64 v_k = m.getCol(k);
        VecF64 temp(m.sizeI());
        for(unsigned int p=0;p<k;p++){
            temp+=productInner(u[p],v_k)/productInner(u[p],u[p])*u[p];
        }
        u(k)=v_k-temp;
    }
    Mat2F64 out(m.sizeI(),m.sizeI());
    for(unsigned int k=0;k<m.sizeI();k++){
        u(k)/=u(k).norm();
        out.setCol(k,u(k));
    }
    return out;

}
void LinearAlgebra::QRDecomposition(const Mat2F64 &m, Mat2F64 &Q, Mat2F64 &R)throw(pexception){
    Q = LinearAlgebra::orthogonalGramSchmidt(m);
    R.clear();
    R.resize(m.sizeI(),m.sizeJ());


    std::vector<VecF64> v_a(m.sizeI(),VecF64(m.sizeI()));
    for(unsigned int j =0;j<m.sizeJ();j++)
        v_a[j]=m.getCol(j);

    for(unsigned int i =0;i<m.sizeI();i++){
        VecF64 e = Q.getCol(i);
        for(unsigned int j =i;j<m.sizeJ();j++){
            R(i,j)=productInner(e,v_a[j]);
        }
    }
}
VecF64 LinearAlgebra::eigenValue(const Mat2x<F64,2,2> &m)throw(pexception){
    double T = m.trace();
    double D = m.determinant();
    double sum = T*T/4 -D;
    VecF64 eigen_value(2);
    if(sum>0)
    {
        sum = std::sqrt(sum);
        eigen_value(0) = T/2 + (sum);
        eigen_value(1) = T/2 - (sum);
        return eigen_value;
    }else{
        return VecF64();
    }

}
VecF64 LinearAlgebra::eigenValue(const Mat2x<F64,3,3> &A,EigenValueMethod method ,F64 error)throw(pexception){


    if(method==Symmetric){
    // Given a real symmetric 3x3 matrix A, compute the eigenvalues

    double p1 = A(0,1)*A(0,1) + A(0,2)*A(0,2) + A(1,2)*A(1,2);
    if (p1 == 0) {
        // A is diagonal.
        VecF64 eig(3);
        eig(0) = A(0,0);
        eig(1) = A(1,1);
        eig(2) = A(2,2);
        return eig;
    }
    else{
        double q = A.trace()/3;
        double p2 = std::pow(A(0,0) - q,2) + std::pow(A(1,1) - q,2) + std::pow(A(2,2) - q,2) + 2 * p1;
        double        p = std::sqrt(p2 / 6);

        Mat2x<F64,3,3>  B = (1 / p)*(A - q*A.identity());//      % I is the identity matrix
        double        r = B.determinant() / 2;

        //In exact arithmetic for a symmetric matrix  -1 <= r <= 1
        //but computation error can leave it slightly outside this range.
        double phi;
        if (r <= -1)
            phi = pop::PI / 3;
        else if (r >= 1)
            phi = 0;
        else
            phi = std::acos(r) / 3;
        VecF64 eig(3);
        eig(0)= q + 2 * p * std::cos(phi);
        eig(2) = q + 2 * p * std::cos(phi + (2*pop::PI/3));
        eig(1) = 3 * q - eig(0) - eig(2) ;   // since trace(A) = eig1 + eig2 + eig3;
        return eig;
    }
    }else{
        return eigenValue(Mat2F64(A),method,error);
    }
}
VecF64 LinearAlgebra::eigenValue(const Mat2F64 &m,EigenValueMethod ,F64 )throw(pexception){
    Mat2F64 M_k(m);
    Mat2F64 Q_k;
    Mat2F64 R_k;

    while(LinearAlgebra::isDiagonal(M_k)==false){
        LinearAlgebra::QRDecomposition(M_k,Q_k,R_k);
        M_k = R_k * Q_k;
    }
    VecF64 v(M_k.sizeI());
    for(unsigned int i=0;i<v.size();i++){
        v(i)=M_k(i,i);
    }
    return v;
}

Mat2F64 LinearAlgebra::solvingLinearSystemGaussianElimination(const Mat2F64 &m){

    Mat2F64 M(m);
    for(unsigned int k =0;k<M.sizeI();k++){
        double valuemax=0;
        int i_max=0;
        //Find pivot for column k:
        for(unsigned int i=k;i<M.sizeI();i++){
            double v = absolute(M(i, k));
            if(v>valuemax){
                i_max = i;
                valuemax=v;
            }

        }
        if( M(i_max, k) == 0){

            //            std::cout<< "Matrix is singular!";
            return M;
        }
        M.swapRow(k,i_max);
        F64 temp = M(k,k);
        for(unsigned int j=0;j<M.sizeJ();j++){
            M(k,j)/=temp;
        }
        for(unsigned int i =0;i<M.sizeI();i++){
            if(i!=k){
                F64 temp = M(i, k) ;
                for(unsigned int j=0;j<M.sizeJ();j++){
                    M(i,j) = M(i, j) - M(k, j) * (temp / M(k, k));
                }
            }
        }
    }
    return M;
}
VecF64 LinearAlgebra::solvingLinearSystemGaussianElimination(const Mat2x22F64 &A,const VecF64 & b)throw(pexception){
    VecF64 v(2);
    double div = 1/(A(0,0)*A(1,1)-A(1,0)*A(0,1));
    v(0)= (b(0)*A(1,1)-b(1)*A(0,1))*div;
    v(1)=-(b(0)*A(1,0)-b(1)*A(0,0))*div;
    return v;
}

VecF64 LinearAlgebra::solvingLinearSystemGaussianElimination(const Mat2F64 &A,const VecF64 & b)throw(pexception)
{
    Mat2F64 M(A);
    VecF64 x(b);
    for(unsigned int k =0;k<M.sizeI();k++){
        F64 temp = M(k,k);
        if(temp==0)
            throw(pexception("Cannot solve this system"));
        for(unsigned int j=0;j<M.sizeJ();j++){
            M(k,j)/=temp;
        }
        x(k)/=temp;
        for(unsigned int i =0;i<M.sizeI();i++){
            if(i!=k){
                F64 temp = M(i, k) ;
                for(unsigned int j= 0;j<M.sizeJ();j++){
                    M(i,j) = M(i, j) - M(k, j) * (temp / M(k, k));
                }
                x(i)= x(i)- x(k) * (temp / M(k, k));
            }
        }

    }
    return x;
}

void solvingLinearSystemGaussianEliminationNonInvertible(Mat2F64 &M)throw(pexception){
    for(unsigned int k =0;k<M.sizeI()-1;k++){

        double valuemax=0;
        int i_max=0;
        //Find pivot for column k:
        for(unsigned int i=k;i<M.sizeI();i++){
            double v = absolute(M(i, k));
            if(v>valuemax){
                i_max = i;
                valuemax=v;
            }

        }
        if( M(i_max, k) == 0)
            std::cout<< "Matrix is singular!";
        M.swapRow(k,i_max);
        F64 temp = M(k,k);
        for(unsigned int j=0;j<M.sizeJ();j++){
            M(k,j)/=temp;
        }
        for(unsigned int i =0;i<M.sizeI();i++){
            if(i!=k){
                F64 temp = M(i, k) ;
                for(unsigned int j=0;j<M.sizeJ();j++){
                    M(i,j) = M(i, j) - M(k, j) * (temp / M(k, k));
                }
            }

        }
    }
}
Mat2F64 LinearAlgebra::eigenVectorGaussianElimination(const Mat2F64 &m,VecF64 v_eigen_value)throw(pexception){
    Mat2F64 I = m.identity(m.sizeI());
    Mat2F64 EigenVec(m.sizeI(),m.sizeI());

    for(unsigned int j =0;j<v_eigen_value.size();j++){
        Mat2F64 m2=      I*  v_eigen_value(j) ;
        Mat2F64 M = m - m2;
        solvingLinearSystemGaussianEliminationNonInvertible(M);
        VecF64 v(m.sizeI());
        v(v.size()-1)=1;
        for(int i=v.size()-2;i>=0;i--){
            v(i) = - M(i,m.sizeJ()-1);
        }
        EigenVec.setCol(j,v);
    }
    return EigenVec;
}


void LinearAlgebra::LUDecomposition(const Mat2F64 &m,Mat2F64 & L,  Mat2F64 & U)throw(pexception){
    if (m.sizeI() != m.sizeJ())
    {
        throw(pexception("In linearAlgebra::LUDecomposition, input Mat2F64 must be a square Mat2F64"));
    }
    L.clear();
    U.clear();
    L.resize(m.sizeI(),m.sizeI());
    U.resize(m.sizeI(),m.sizeI());
    for (unsigned int i=0;i<m.sizeI();i++)
    {
        L(i,i)=1;
        for(unsigned int j=i;j<m.sizeI();j++) {
            F64 sum=0;
            for(unsigned int s=0;s<=i-1;s++) {
                sum+= L(i,s)*U(s,j);
            }
            U(i,j)=m(i,j)-sum;
        }
        for(unsigned int k=i+1;k<m.sizeI();k++) {
            F64 sum=0;
            for(unsigned int s=0;s<=i-1;s++) {
                sum+=L(k,s)*U(s,i);
            }
            L(k,i)= (m(k,i)-sum)/U(i,i);
        }
    }
}
Mat2F64 LinearAlgebra::AATransposeEqualMDecomposition(const Mat2F64 &M)throw(pexception){
    Mat2F64 A,U;
    LinearAlgebra::LUDecomposition(M,A, U);
    for(unsigned int j =0;j<M.sizeI();j++)
        for(unsigned int i =j;i<M.sizeI();i++)
            A(i,j)*=std::sqrt(U(j,j));

    return A;
}
Vec2F64  LinearAlgebra::linearLeastSquares(const Mat2F64 &X,const VecF64& Y){
    Mat2F64 Xtranspose=X.transpose();
    Mat2F64 M(Xtranspose*X);
    return M.inverse()*(Xtranspose*Y);
}

}
