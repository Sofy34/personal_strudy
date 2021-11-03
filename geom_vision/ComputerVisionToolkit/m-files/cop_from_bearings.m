function C = cop_from_bearings(U,B)
%COP_FROM_BEARINGS  Camera locations from bearings
   
    B(1,:) = []; % remove node 1
    
    S=kron(eye(size(B,2)),ones(3,3));
    for i=1:size(U,2)
        S(3*i-2:3 *i,3*i-2:3*i) = skew(U(:,i)) ;
    end
    
    F=S*kron(B',eye(3));
    [~,~,V] = svd(F);
    X =  reshape(V(:,end),3,[]);
    s = sign(U(:,1)'*X*B(:,1)); % fix sign  
    X = s * [[0;0;0],X]; % add node 1

    C = num2cell(reshape(X,3,[]),[1,size(X,2)]);    
end
