
-- Vertex

in vec4 Position;
out int vInstance;

void main()
{
    gl_Position = Position;
    vInstance = gl_InstanceID;
}

-- Fill

out float FragColor;

void main()
{
    FragColor = 1.0f;
}

-- PickLayer

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;
 
in int vInstance[3];
out float gLayer;
 
void main()
{
    gl_Layer = vInstance[0];
    gLayer = float(gl_Layer) + 0.5;
    gl_Position = gl_in[0].gl_Position;
    EmitVertex();
    gl_Position = gl_in[1].gl_Position;
    EmitVertex();
    gl_Position = gl_in[2].gl_Position;
    EmitVertex();
    EndPrimitive();
}

-- Advect

out vec4 FragColor;

uniform sampler3D VelocityTexture;
uniform sampler3D SourceTexture;
uniform sampler3D Obstacles;
uniform sampler3D ObstacleSpeeds;

uniform vec3 InverseSize;
uniform float TimeStep;
uniform float Dissipation;

in float gLayer;

void main()
{
    vec3 fragCoord = vec3(gl_FragCoord.xy, gLayer);
    float solid = texture(Obstacles, InverseSize * fragCoord).x;
    if (solid > 0) {
        FragColor = vec4(0);//texture(ObstacleSpeeds, InverseSize * fragCoord);//vec4(0);
        return;
    }

    vec3 u = texture(VelocityTexture, InverseSize * fragCoord).xyz; //InverseSize is needed to sample from normalized texture coordinates

    vec3 coord = InverseSize * (fragCoord - TimeStep * u);
    FragColor = Dissipation * texture(SourceTexture, coord);
}

-- Jacobi

out vec4 FragColor;

uniform sampler3D Pressure;
uniform sampler3D Divergence;
uniform sampler3D Obstacles;
uniform sampler3D ObstacleSpeeds;

uniform float Alpha;
uniform float InverseBeta;

in float gLayer;

void main()
{
    //why doesn't this handle occupied cells? (might be handled in SubtractDivergence)
    ivec3 T = ivec3(gl_FragCoord.xy, gLayer);

    // Find neighboring pressure:
    vec4 pN = texelFetchOffset(Pressure, T, 0, ivec3(0, 1, 0));
    vec4 pS = texelFetchOffset(Pressure, T, 0, ivec3(0, -1, 0));
    vec4 pE = texelFetchOffset(Pressure, T, 0, ivec3(1, 0, 0));
    vec4 pW = texelFetchOffset(Pressure, T, 0, ivec3(-1, 0, 0));
    vec4 pU = texelFetchOffset(Pressure, T, 0, ivec3(0, 0, 1));
    vec4 pD = texelFetchOffset(Pressure, T, 0, ivec3(0, 0, -1));
    vec4 pC = texelFetch(Pressure, T, 0);

    // Find neighboring obstacles:
    float oN = texelFetchOffset(Obstacles, T, 0, ivec3(0, 1, 0)).r;//.xyz;
    float oS = texelFetchOffset(Obstacles, T, 0, ivec3(0, -1, 0)).r;//.xyz;
    float oE = texelFetchOffset(Obstacles, T, 0, ivec3(1, 0, 0)).r;//.xyz;
    float oW = texelFetchOffset(Obstacles, T, 0, ivec3(-1, 0, 0)).r;//.xyz;
    float oU = texelFetchOffset(Obstacles, T, 0, ivec3(0, 0, 1)).r;//.xyz;
    float oD = texelFetchOffset(Obstacles, T, 0, ivec3(0, 0, -1)).r;//.xyz;

    //the following might need modification for
    // Use center pressure for solid cells:
    if (oN > 0) pN = pC;
    if (oS > 0) pS = pC;
    if (oE > 0) pE = pC;
    if (oW > 0) pW = pC;
    if (oU > 0) pU = pC;
    if (oD > 0) pD = pC;

    vec4 bC = texelFetch(Divergence, T, 0);
    FragColor = (pW + pE + pS + pN + pU + pD + Alpha * bC) * InverseBeta;
}

-- SubtractGradient

out vec3 FragColor;

uniform sampler3D Velocity;
uniform sampler3D Pressure;
uniform sampler3D Obstacles;
uniform sampler3D ObstacleSpeeds;
uniform float GradientScale;

in float gLayer;

void main()
{
    ivec3 T = ivec3(gl_FragCoord.xy, gLayer);

    float oC = texelFetch(Obstacles, T, 0).r;
    vec3 osC = texelFetch(Obstacles, T, 0).xyz;
    if (oC > 0) {
        FragColor = osC;
        return;
    }

    // Find neighboring pressure:
    float pN = texelFetchOffset(Pressure, T, 0, ivec3(0, 1, 0)).r;
    float pS = texelFetchOffset(Pressure, T, 0, ivec3(0, -1, 0)).r;
    float pE = texelFetchOffset(Pressure, T, 0, ivec3(1, 0, 0)).r;
    float pW = texelFetchOffset(Pressure, T, 0, ivec3(-1, 0, 0)).r;
    float pU = texelFetchOffset(Pressure, T, 0, ivec3(0, 0, 1)).r;
    float pD = texelFetchOffset(Pressure, T, 0, ivec3(0, 0, -1)).r;
    float pC = texelFetch(Pressure, T, 0).r;

    //might need to add a .r at the end of the RHS here
    // Find neighboring obstacles:
    float oN = texelFetchOffset(Obstacles, T, 0, ivec3(0, 1, 0)).r;//.xyz;
    float oS = texelFetchOffset(Obstacles, T, 0, ivec3(0, -1, 0)).r;//.xyz;
    float oE = texelFetchOffset(Obstacles, T, 0, ivec3(1, 0, 0)).r;//.xyz;
    float oW = texelFetchOffset(Obstacles, T, 0, ivec3(-1, 0, 0)).r;//.xyz;
    float oU = texelFetchOffset(Obstacles, T, 0, ivec3(0, 0, 1)).r;//.xyz;
    float oD = texelFetchOffset(Obstacles, T, 0, ivec3(0, 0, -1)).r;//.xyz;

    // Find neighboring obstacle speeds
    vec3 osN = texelFetchOffset(ObstacleSpeeds, T, 0, ivec3(0, 1, 0)).xyz;
    vec3 osS = texelFetchOffset(ObstacleSpeeds, T, 0, ivec3(0, -1, 0)).xyz;
    vec3 osE = texelFetchOffset(ObstacleSpeeds, T, 0, ivec3(1, 0, 0)).xyz;
    vec3 osW = texelFetchOffset(ObstacleSpeeds, T, 0, ivec3(-1, 0, 0)).xyz;
    vec3 osU = texelFetchOffset(ObstacleSpeeds, T, 0, ivec3(0, 0, 1)).xyz;
    vec3 osD = texelFetchOffset(ObstacleSpeeds, T, 0, ivec3(0, 0, -1)).xyz;

    // Use center pressure for solid cells:
    vec3 obstV = vec3(0);
    vec3 vMask = vec3(1);

    if (oN > 0) { pN = pC; obstV.y = osN.y; vMask.y = 0; }
    if (oS > 0) { pS = pC; obstV.y = osS.y; vMask.y = 0; }
    if (oE > 0) { pE = pC; obstV.x = osE.x; vMask.x = 0; }
    if (oW > 0) { pW = pC; obstV.x = osW.x; vMask.x = 0; }
    if (oU > 0) { pU = pC; obstV.z = osU.z; vMask.z = 0; }
    if (oD > 0) { pD = pC; obstV.z = osD.z; vMask.z = 0; }

    // Enforce the free-slip boundary condition:
    vec3 oldV = texelFetch(Velocity, T, 0).xyz;
    vec3 grad = vec3(pE - pW, pN - pS, pU - pD) * GradientScale;
    vec3 newV = oldV - grad;
    FragColor = (vMask * newV) + obstV; //newV; //obstV will be completely 0 for static obstacles
}

-- ComputeDivergence

out float FragColor;

uniform sampler3D Velocity;
uniform sampler3D Obstacles;
uniform sampler3D ObstacleSpeeds;
uniform float HalfInverseCellSize;

in float gLayer;

void main()
{
    ivec3 T = ivec3(gl_FragCoord.xy, gLayer);

    // Find neighboring velocities:
    vec3 vN = texelFetchOffset(Velocity, T, 0, ivec3(0, 1, 0)).xyz;
    vec3 vS = texelFetchOffset(Velocity, T, 0, ivec3(0, -1, 0)).xyz;
    vec3 vE = texelFetchOffset(Velocity, T, 0, ivec3(1, 0, 0)).xyz;
    vec3 vW = texelFetchOffset(Velocity, T, 0, ivec3(-1, 0, 0)).xyz;
    vec3 vU = texelFetchOffset(Velocity, T, 0, ivec3(0, 0, 1)).xyz;
    vec3 vD = texelFetchOffset(Velocity, T, 0, ivec3(0, 0, -1)).xyz;

    // Find neighboring ObstacleSpeeds:
    vec3 osN = texelFetchOffset(ObstacleSpeeds, T, 0, ivec3(0, 1, 0)).xyz;
    vec3 osS = texelFetchOffset(ObstacleSpeeds, T, 0, ivec3(0, -1, 0)).xyz;
    vec3 osE = texelFetchOffset(ObstacleSpeeds, T, 0, ivec3(1, 0, 0)).xyz;
    vec3 osW = texelFetchOffset(ObstacleSpeeds, T, 0, ivec3(-1, 0, 0)).xyz;
    vec3 osU = texelFetchOffset(ObstacleSpeeds, T, 0, ivec3(0, 0, 1)).xyz;
    vec3 osD = texelFetchOffset(ObstacleSpeeds, T, 0, ivec3(0, 0, -1)).xyz;

    float oN = texelFetchOffset(Obstacles, T, 0, ivec3(0, 1, 0)).r;
    float oS = texelFetchOffset(Obstacles, T, 0, ivec3(0, -1, 0)).r;
    float oE = texelFetchOffset(Obstacles, T, 0, ivec3(1, 0, 0)).r;
    float oW = texelFetchOffset(Obstacles, T, 0, ivec3(-1, 0, 0)).r;
    float oU = texelFetchOffset(Obstacles, T, 0, ivec3(0, 0, 1)).r;
    float oD = texelFetchOffset(Obstacles, T, 0, ivec3(0, 0, -1)).r;

    // Use obstacle velocities for solid cells:
    if (oN > 0) vN = osN.xyz; //does not account for osbject speed
    if (oS > 0) vS = osS.xyz;
    if (oE > 0) vE = osE.xyz;
    if (oW > 0) vW = osW.xyz;
    if (oU > 0) vU = osU.xyz;
    if (oD > 0) vD = osD.xyz;

    //may need to change back to this
    //FragColor = HalfInverseCellSize * (vE.x - vW.x + vN.y - vS.y + vU.z - vD.z);
    FragColor = HalfInverseCellSize * (vE.x - vW.x + vN.y - vS.y + vU.z - vD.z);
}

-- Splat

out vec4 FragColor;

uniform vec3 Point;
uniform float Radius;
uniform vec3 FillColor;
uniform float Jitter;
uniform float Time;

in float gLayer;

float rand(vec3 co){
    return fract(sin(dot(co, vec3(12.9898,78.233,34.3948))) * 43758.5453 * Time);
}

void main()
{
    vec3 here = vec3(gl_FragCoord.xy, gLayer);
    float d = distance(Point, here);
    if (d < Radius) {
        float a = (Radius - d) * 0.5;
        a = min(a, 1.0);
        FragColor = vec4(FillColor + Jitter * (rand(here) - 0.5), a);
    } else {
        FragColor = vec4(0);
    }
}

-- Buoyancy

out vec3 FragColor;
uniform sampler3D Velocity;
uniform sampler3D Temperature;
uniform sampler3D Density;
uniform float AmbientTemperature;
uniform float TimeStep;
uniform float Sigma;
uniform float Kappa;

in float gLayer;

void main()
{
    ivec3 TC = ivec3(gl_FragCoord.xy, gLayer);
    float T = texelFetch(Temperature, TC, 0).r;
    vec3 V = texelFetch(Velocity, TC, 0).xyz;

    FragColor = V;

    if (T > AmbientTemperature) {
        float D = texelFetch(Density, TC, 0).x;
        FragColor += (TimeStep * (T - AmbientTemperature) * Sigma - D * Kappa ) * vec3(0, 0, 1);
    }
}

--UpdateObstacles

out float FragColor;
//uniform sampler3D Obstacles;
//uniform int CurrentTime;
uniform int StartX;
uniform int EndX;
uniform int StartY;
uniform int EndY;
uniform int StartZ;
uniform int EndZ;

in float gLayer;

void main()
{
    ivec3 TC = ivec3(gl_FragCoord.xy, gLayer);
    //float middleX = sin(Speed * CurrentTime);

    //if(abs(middleX - TC.x) < CubeSizeX/2.0*100 && abs(TC.y) < CubeSizeY/2.0*100 && abs(TC.z) < CubeSizeZ/2.0*100)
    //if(TC.x<40 && TC.y<40 && TC.z<40)
    if(TC.x >= StartX && TC.x < EndX && TC.y >= StartY && TC.y<EndY && TC.z>=StartZ && TC.z<EndZ)
        FragColor = 1.0;
    else
        FragColor = 0.0;
}

--UpdateObstacleSpeeds

out vec3 FragColor;
//uniform sampler3D ObstacleSpeeds;
uniform float CurrentTime;
uniform float CubeSizeX;
uniform float CubeSizeY;
uniform float CubeSizeZ;
uniform float Speed;
uniform float Amplitude;

in float gLayer;

void main()
{
    ivec3 TC = ivec3(gl_FragCoord.xy, gLayer);
    float XSpeed = Amplitude * cos(Speed * CurrentTime);

    FragColor = vec3(XSpeed, 0.0, 0.0);
}

--ApplyObstacleSpeeds

out vec3 FragColor;
uniform sampler3D Velocity;
uniform sampler3D Obstacles;
uniform sampler3D ObstacleSpeeds;

in float gLayer;

void main()
{
    ivec3 TC = ivec3(gl_FragCoord.xy, gLayer);
    float solid = texelFetch(Obstacles, TC, 0).x;//texture(Obstacles, TC, 0).x;
    if(solid > 0)
        FragColor = texelFetch(ObstacleSpeeds, TC, 0).xyz;//texture(ObstacleSpeeds, TC, 0).xyz;
    else
        FragColor = texelFetch(Velocity, TC, 0).xyz;//texture(Velocity, TC, 0).xyz;
}

--CopyVelocity
out vec3 FragColor;
uniform sampler3D Velocity;

in float gLayer;

void main(){
    ivec3 T = ivec3(gl_FragCoord.xy, gLayer);
    FragColor = texelFetch(Velocity, T, 0).xyz;
}

--VorticityConfinement

out vec3 FragColor;
uniform sampler3D Velocity;
uniform float DoubleCellSize;
uniform float VorticityBoost;
uniform float TimeStep;
uniform ivec3 VelocityTextureSize;

in float gLayer;

void main()
{
    ivec3 T = ivec3(gl_FragCoord.xy, gLayer);

    vec3 vC = texelFetch(Velocity, T, 0).xyz;

    //check if we are at edges of texture here; do nothing at these points to avoid out-of-bounds
    if(T.x == 0 || T.y == 0 || T.z == 0 || T.x == VelocityTextureSize.x-1 || T.y == VelocityTextureSize.y-1 || T.z == VelocityTextureSize.z-1){
        FragColor = vC;
        return;}

    // Find neighboring velocities:
    vec3 vN = texelFetchOffset(Velocity, T, 0, ivec3(0, 1, 0)).xyz;
    vec3 vS = texelFetchOffset(Velocity, T, 0, ivec3(0, -1, 0)).xyz;
    vec3 vE = texelFetchOffset(Velocity, T, 0, ivec3(1, 0, 0)).xyz;
    vec3 vW = texelFetchOffset(Velocity, T, 0, ivec3(-1, 0, 0)).xyz;
    vec3 vU = texelFetchOffset(Velocity, T, 0, ivec3(0, 0, 1)).xyz;
    vec3 vD = texelFetchOffset(Velocity, T, 0, ivec3(0, 0, -1)).xyz;

    vec3 curl;

    curl.x = ((vN.z - vS.z) - (vU.y - vD.y));
    curl.y = ((vU.x - vD.x) - (vE.z - vW.z));
    curl.z = ((vE.y - vW.y) - (vN.x - vS.x));

    curl /= DoubleCellSize;

    vec3 gradNorm;

    gradNorm.x = length(vE) - length(vW);
    gradNorm.y = length(vN) - length(vS);
    gradNorm.z = length(vU) - length(vD);

    gradNorm /= DoubleCellSize;

    vec3 N; //gradNorm normalized

    N = (gradNorm.x == 0 || gradNorm.y == 0 || gradNorm.z == 0) ? vec3(0) : normalize(gradNorm);

    //the force VorticityBoost * DoubleCellSize / 2 * cross(N, curl) is multiplied by the timestep and added to the current velocity value
    FragColor = vC + TimeStep * VorticityBoost * DoubleCellSize / 2 * cross(N, curl);
}

/*__global__ void GetVorticityConfinementForce(
    CudaFlagGrid flags, CudaVecGrid curl, CudaRealGrid curl_norm,
    const float strength, CudaVecGrid force, const int32_t bnd) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }

  if (i < bnd || i > flags.xsize() - 1 - bnd ||
      j < bnd || j > flags.ysize() - 1 - bnd ||
      (flags.is_3d() && (k < bnd || k > flags.zsize() - 1 - bnd))) {
    // Don't add force on the boundaries.
    force.setSafe(i, j, k, b, CudaVec3(0, 0, 0));
    return;
  }

  CudaVec3 grad(0, 0, 0);
  grad.x = 0.5f * (curl_norm(i + 1, j, k, b) - curl_norm(i - 1, j, k, b));
  grad.y = 0.5f * (curl_norm(i, j + 1, k, b) - curl_norm(i, j - 1, k, b));
  if (flags.is_3d()) {
    grad.z = 0.5f * (curl_norm(i, j, k + 1, b) - curl_norm(i, j, k - 1, b));
  }
  grad.normalize();

  force.setSafe(i, j, k, b, CudaVec3::cross(grad, curl(i, j, k, b)) * strength);
}*/
