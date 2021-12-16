
-- Vertex

in vec4 Position;
out int vInstance;

void main()
{
    gl_Position = Position;
    vInstance = gl_InstanceID;
}

-- Fill

out vec3 FragColor;

void main()
{
    FragColor = vec3(1, 0, 0);
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

-- AdvectVelocity

//corrected I think

out vec4 FragColor;

uniform sampler3D VelocityTexture;
uniform sampler3D SourceTexture;
uniform sampler3D Obstacles;

uniform vec3 InverseSize; //this is the inverse of the size of the velocity
uniform float TimeStep;
uniform float Dissipation;
uniform vec3 InverseSizeVelocity;
//uniform vec3 SizeRatio; //=ObstacleTextureSize / VelocityTextureSize

in float gLayer;

void main()
{
    vec3 fragCoord = vec3(gl_FragCoord.xy, gLayer);
    float solid = texture(Obstacles, InverseSize * (fragCoord - 0.5)).x; //need to use OBSTACLE coordinates, FragColor uses VELOCITY coordinates
    if (solid > 0) { //a value of 0.5 might be worth considering; it would put the cutoff value at the edge of the cell, but we probably want the edge velocity to be equal to the obstacle velocity
        FragColor = vec4(0);
        return;
    }

    vec3 u = texture(VelocityTexture, InverseSizeVelocity * fragCoord).xyz; //probably correct

    vec3 coord = InverseSize * (fragCoord - TimeStep * u);
    FragColor = Dissipation * texture(SourceTexture, coord);
}

-- Advect

//corrected I think

out vec4 FragColor;

uniform sampler3D VelocityTexture;
uniform sampler3D SourceTexture;
uniform sampler3D Obstacles;

uniform vec3 InverseSize; //this is the inverse of the size of the ADVECTED PROPERTY (so NOT the velocity)
uniform float TimeStep;
uniform float Dissipation;
uniform vec3 InverseSizeVelocity;

//uniform vec3 InverseSizeRatio;  //=VelocityTextureSize / ObstacleTextureSize

in float gLayer;

void main()
{
    vec3 fragCoord = vec3(gl_FragCoord.xy, gLayer);
    float solid = texture(Obstacles, InverseSize * fragCoord).x;
    if (solid > 0) {
        FragColor = vec4(0);
        return;
    }

    vec3 u = texture(VelocityTexture, InverseSizeVelocity * (fragCoord + 0.5)).xyz;

    /*
    ivec3 T = ivec3(gl_FragCoord.xy, gLayer);
    vec3 vThis = texelFetch(VelocityTexture, T, 0).xyz;
    float vD = texelFetchOffset(VelocityTexture, T, 0, ivec3(0,-1,0)).y;
    float vW = texelFetchOffset(VelocityTexture, T, 0, ivec3(-1,0,0)).x;
    float vS = texelFetchOffset(VelocityTexture, T, 0, ivec3(0,0,-1)).z;

    vec3 u;

    u.x = (vE + vThis.x) / 2;
    u.y = (vU + vThis.y) / 2;
    u.z = (vN + vThis.z) / 2;
    */

    vec3 coord = InverseSize * (fragCoord - TimeStep * u);
    FragColor = Dissipation * texture(SourceTexture, coord);
}

-- Jacobi

//NOT corrected -------------------------

out vec4 FragColor;

uniform sampler3D Pressure;
uniform sampler3D Divergence;
uniform sampler3D Obstacles;

uniform float Alpha;
uniform float InverseBeta;

in float gLayer;

void main()
{
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
    vec3 oN = texelFetchOffset(Obstacles, T, 0, ivec3(0, 1, 0)).xyz;
    vec3 oS = texelFetchOffset(Obstacles, T, 0, ivec3(0, -1, 0)).xyz;
    vec3 oE = texelFetchOffset(Obstacles, T, 0, ivec3(1, 0, 0)).xyz;
    vec3 oW = texelFetchOffset(Obstacles, T, 0, ivec3(-1, 0, 0)).xyz;
    vec3 oU = texelFetchOffset(Obstacles, T, 0, ivec3(0, 0, 1)).xyz;
    vec3 oD = texelFetchOffset(Obstacles, T, 0, ivec3(0, 0, -1)).xyz;

    // Use center pressure for solid cells:
    if (oN.x > 0) pN = pC;
    if (oS.x > 0) pS = pC;
    if (oE.x > 0) pE = pC;
    if (oW.x > 0) pW = pC;
    if (oU.x > 0) pU = pC;
    if (oD.x > 0) pD = pC;

    vec4 bC = texelFetch(Divergence, T, 0);
    FragColor = (pW + pE + pS + pN + pU + pD + Alpha * bC) * InverseBeta;
}

-- SubtractGradient

//corrected?

//Jacobi was before this one
out vec3 FragColor;

uniform sampler3D Velocity;
uniform sampler3D Pressure;
uniform sampler3D Obstacles;
uniform float GradientScale;

in float gLayer;

void main()
{
    ivec3 T = ivec3(gl_FragCoord.xy, gLayer);

    vec3 oC = texelFetch(Obstacles, T, 0).xyz;

    //Find neighboring obstacles
    vec3 oS = texelFetchOffset(Obstacles, T, 0, ivec3(0, -1, 0)).xyz;
    vec3 oW = texelFetchOffset(Obstacles, T, 0, ivec3(-1, 0, 0)).xyz;
    vec3 oD = texelFetchOffset(Obstacles, T, 0, ivec3(0, 0, -1)).xyz;

    if (oC.x > 0) {
        FragColor = oC.yzx; //???
        return;
    }

    // Find neighboring pressure:
    float pS = texelFetchOffset(Pressure, T, 0, ivec3(0, -1, 0)).r;
    float pW = texelFetchOffset(Pressure, T, 0, ivec3(-1, 0, 0)).r;
    float pD = texelFetchOffset(Pressure, T, 0, ivec3(0, 0, -1)).r;
    float pC = texelFetch(Pressure, T, 0).r;

    // Use center pressure for solid cells:
    vec3 obstV = vec3(0);
    vec3 vMask = vec3(1);

    if (oS.x > 0) { pS = pC; obstV.y = oS.z; vMask.y = 0; }
    if (oW.x > 0) { pW = pC; obstV.x = oW.y; vMask.x = 0; }
    if (oD.x > 0) { pD = pC; obstV.z = oD.x; vMask.z = 0; }

    // Enforce the free-slip boundary condition:
    vec3 oldV = texelFetch(Velocity, T, 0).xyz;
    //vec3 grad = vec3(pE - pW, pN - pS, pU - pD) * GradientScale; //GradientScale was originally 2 times larger
    vec3 grad = vec3(pC - pW, pC - pS, pC - pD) * GradientScale / 2;
    vec3 newV = oldV - grad;
    FragColor = (vMask * newV) + obstV;
    //FragColor = oldV;
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

    // these were for non-staggered
    //vec3 vN = texelFetchOffset(Velocity, T, 0, ivec3(0, 1, 0)).xyz;
    //vec3 vS = texelFetchOffset(Velocity, T, 0, ivec3(0, -1, 0)).xyz;
    //vec3 vE = texelFetchOffset(Velocity, T, 0, ivec3(1, 0, 0)).xyz;
    //vec3 vW = texelFetchOffset(Velocity, T, 0, ivec3(-1, 0, 0)).xyz;
    //vec3 vU = texelFetchOffset(Velocity, T, 0, ivec3(0, 0, 1)).xyz;
    //vec3 vD = texelFetchOffset(Velocity, T, 0, ivec3(0, 0, -1)).xyz;

    //these are for staggered
    vec3 vN = texelFetchOffset(Velocity, T, 0, ivec3(0, 1, 0)).xyz;
    vec3 vThis = texelFetch(Velocity, T, 0).xyz;
    vec3 vE = texelFetchOffset(Velocity, T, 0, ivec3(1, 0, 0)).xyz;
    vec3 vU = texelFetchOffset(Velocity, T, 0, ivec3(0, 0, 1)).xyz;

    // Find neighboring obstacles:
    vec3 oN = texelFetchOffset(Obstacles, T, 0, ivec3(0, 1, 0)).xyz;
    //vec3 oS = texelFetchOffset(Obstacles, T, 0, ivec3(0, -1, 0)).xyz;
    vec3 oE = texelFetchOffset(Obstacles, T, 0, ivec3(1, 0, 0)).xyz;
    //vec3 oW = texelFetchOffset(Obstacles, T, 0, ivec3(-1, 0, 0)).xyz;
    vec3 oU = texelFetchOffset(Obstacles, T, 0, ivec3(0, 0, 1)).xyz;
    //vec3 oD = texelFetchOffset(Obstacles, T, 0, ivec3(0, 0, -1)).xyz;
    vec3 oThis = texelFetch(Obstacles, T, 0).xyz;

    // Use obstacle velocities for solid cells:
    if (oN.x > 0) vN = oN.yzx;
    if (oThis.x > 0) vThis = oThis.yzx;
    if (oE.x > 0) vE = oE.yzx;
    //if (oW.x > 0) vW = oW.yzx;
    if (oU.x > 0) vU = oU.yzx;
    //if (oD.x > 0) vD = oD.yzx;

    //FragColor = HalfInverseCellSize * (vE.x - vW.x + vN.y - vS.y + vU.z - vD.z);

    FragColor = HalfInverseCellSize * 2 * (vE.x - vThis.x + vN.y - vThis.y + vU.z - vThis.z); //CHANGE TO INVERSECELLSIZE, SUBSEQUENT INDICES ARE ONE CELL WIDTH APART FROM EACH OTHER
}

-- Splat

//this is an inflow, called separately for temperature and density
//ok, although seems to ignore whatever temperature/density is already in the splat before impulse is applied

out vec4 FragColor;

uniform vec3 Point;
uniform float Radius;
uniform vec3 FillColor;

in float gLayer;

void main()
{
    float d = distance(Point, vec3(gl_FragCoord.xy, gLayer));
    if (d < Radius) {
        float a = (Radius - d) * 0.5;
        a = min(a, 1.0);
        FragColor = vec4(FillColor, a);
    } else {
        FragColor = vec4(0);
    }
}

-- Buoyancy

//appears correct

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
    //this only supports +y-direction buoyancy

    ivec3 TC = ivec3(gl_FragCoord.xy, gLayer);
    float TUpper = texelFetch(Temperature, TC, 0).r;
    float TLower = texelFetchOffset(Temperature, TC, 0, ivec3(0, -1, 0)).r;

    float T = (TUpper + TLower) / 2;

    vec3 V = texelFetch(Velocity, TC, 0).xyz;

    FragColor = V;

    if (T > AmbientTemperature) {
        float D = texelFetch(Density, TC, 0).x;
        FragColor += (TimeStep * (T - AmbientTemperature) * Sigma - D * Kappa ) * vec3(0, -1, 0);
    }
}
