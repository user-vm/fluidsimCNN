-- VS

in vec4 Position;
out vec4 vPosition;
uniform mat4 ModelviewProjection;

void main()
{
    gl_Position = ModelviewProjection * Position;
    vPosition = Position;
}

-- GS

layout(points) in;
layout(triangle_strip, max_vertices = 24) out;

in vec4 vPosition[1];

uniform mat4 ModelviewProjection;
uniform mat4 ProjectionMatrix;
uniform mat4 ViewMatrix;
uniform mat4 Modelview;

vec4 objCube[8]; // Object space coordinate of cube corner
vec4 ndcCube[8]; // Normalized device coordinate of cube corner
ivec4 faces[6];  // Vertex indices of the cube faces

void emit_vert(int vert)
{
    gl_Position = ndcCube[vert];
    EmitVertex();
}

void emit_face(int face)
{
    emit_vert(faces[face][1]); emit_vert(faces[face][0]);
    emit_vert(faces[face][3]); emit_vert(faces[face][2]);
    EndPrimitive();
}

void main()
{
    faces[0] = ivec4(0,1,3,2); faces[1] = ivec4(5,4,6,7);
    faces[2] = ivec4(4,5,0,1); faces[3] = ivec4(3,2,7,6);
    faces[4] = ivec4(0,3,4,7); faces[5] = ivec4(2,1,6,5);

    vec4 P = vPosition[0];
    vec4 I = vec4(1,0,0,0);
    vec4 J = vec4(0,1,0,0);
    vec4 K = vec4(0,0,1,0);

    objCube[0] = P+K+I+J; objCube[1] = P+K+I-J;
    objCube[2] = P+K-I-J; objCube[3] = P+K-I+J;
    objCube[4] = P-K+I+J; objCube[5] = P-K+I-J;
    objCube[6] = P-K-I-J; objCube[7] = P-K-I+J;

    // Transform the corners of the box:
    for (int vert = 0; vert < 8; vert++)
        ndcCube[vert] = ModelviewProjection * objCube[vert];

    // Emit the six faces:
    for (int face = 0; face < 6; face++)
        emit_face(face);
}

-- FS

out vec4 FragColor;

uniform sampler3D Density;
uniform sampler3D LightCache;
uniform sampler3D Obstacles;

uniform vec3 LightPosition = vec3(1.0, 1.0, 2.0);
uniform vec3 LightIntensity = vec3(10.0);
uniform float Absorption = 10.0;
uniform mat4 Modelview;
uniform float FocalLength;
uniform vec2 WindowSize;
uniform vec3 RayOrigin;
uniform vec3 Ambient = vec3(0.15, 0.15, 0.20);
uniform float StepSize;
uniform int ViewSamples;
uniform vec3 obstacleColor;

const bool Jitter = false;

float GetDensity(vec3 pos)
{

    return texture(Density, pos).x;
}

float GetObstacle(vec3 pos)
{
    //if ((texture(Obstacles, pos).x > 0 || texture(Obstacles, pos).y > 0 || texture(Obstacles, pos).z > 0) &&
    //    pos.x > 1.0/40.0 && pos.y > 1.0/40.0 && pos.z > 1.0/40.0 && pos.x<1.0-1.0/40.0 && pos.y<1.0-1.0/40.0 && pos.z<1.0-1.0/40.0)
    if(texture(Obstacles, pos).x > 0)
        return 200.0;
    else
        return 0.0;
}

struct Ray {
    vec3 Origin;
    vec3 Dir;
};

struct AABB {
    vec3 Min;
    vec3 Max;
};

bool IntersectBox(Ray r, AABB aabb, out float t0, out float t1)
{
    vec3 invR = 1.0 / r.Dir;
    vec3 tbot = invR * (aabb.Min-r.Origin);
    vec3 ttop = invR * (aabb.Max-r.Origin);
    vec3 tmin = min(ttop, tbot);
    vec3 tmax = max(ttop, tbot);
    vec2 t = max(tmin.xx, tmin.yz);
    t0 = max(t.x, t.y);
    t = min(tmax.xx, tmax.yz);
    t1 = min(t.x, t.y);
    return t0 <= t1;
}

float randhash(uint seed, float b)
{
    const float InverseMaxInt = 1.0 / 4294967295.0;
    uint i=(seed^12345391u)*2654435769u;
    i^=(i<<6u)^(i>>26u);
    i*=2654435769u;
    i+=(i<<5u)^(i>>12u);
    return float(b * i) * InverseMaxInt;
}

void main()
{
    //vec3 obstacleColor;
    vec3 rayDirection;
    bool obstacleFound = false;
    rayDirection.xy = 2.0 * gl_FragCoord.xy / WindowSize - 1.0;
    rayDirection.x /= WindowSize.y / WindowSize.x;
    rayDirection.z = -FocalLength;
    rayDirection = (vec4(rayDirection, 0) * Modelview).xyz;

    Ray eye = Ray( RayOrigin, normalize(rayDirection) );
    AABB aabb = AABB(vec3(-1), vec3(1));

    float tnear, tfar;
    IntersectBox(eye, aabb, tnear, tfar);
    if (tnear < 0.001) tnear = 0.001;

    vec3 rayStart = eye.Origin + eye.Dir * tnear;
    vec3 rayStop = eye.Origin + eye.Dir * tfar;
    rayStart = 0.5 * (rayStart + 1.0);
    rayStop = 0.5 * (rayStop + 1.0);

    vec3 pos = rayStart;
    vec3 viewDir = normalize(rayStop-rayStart) * StepSize;
    float T = 1.0;
    vec3 Lo = Ambient;

    if (Jitter) {
        uint seed = uint(gl_FragCoord.x) * uint(gl_FragCoord.y);
        pos += viewDir * (-0.5 + randhash(seed, 1.0));
    }

    float remainingLength = distance(rayStop, rayStart);

    //this goes front to back, not back to front
    for (int i=0; i < ViewSamples && remainingLength > 0.0;
        ++i, pos += viewDir, remainingLength -= StepSize) {

        float density = GetDensity(pos);
        float obstacle = GetObstacle(pos); //GetObstacle should also check for closeness to boundaries, and ignore edge walls
        vec3 lightColor = vec3(1);
        if (pos.z < 0.1) {
            density = 10;
            lightColor = 3*Ambient;
        } else if (density <= 0.01 && obstacle==0.0) {
            //Lo += obstacle * obstacleColor;
            continue;
        }

        if(obstacle > 0.0){
            //T = 1.0;
            //Lo = obstacleColor;
            obstacleFound = true;
            break;
        }
        else
            T *= 1.0 - density * StepSize * Absorption;
        //if (T <= 0.01)
        //    break;

        vec3 Li = lightColor * texture(LightCache, pos).xxx;
        Lo += Li * T * density * StepSize;
    }

    //Lo = 1-Lo;
    if(obstacleFound){
        FragColor.rgb = Lo*(1-T) + T*obstacleColor;
        //FragColor.rgb = Lo*(1-T) + T*(1.0-obstacleColor); //for lighter ground but inverted smoke (uncomment bottom too)
        FragColor.a = 1.0;}
    else{

    FragColor.rgb = Lo;
    FragColor.a = 1-T;}

    FragColor.rgb = FragColor.rgb*FragColor.a+vec3(1,1,1)*(1.0-FragColor.a);
    //FragColor.rgb = 1.0-(FragColor.rgb*FragColor.a+vec3(0,0,0)*(1.0-FragColor.a)); //for lighter ground but inverted smoke (uncomment top too)
    FragColor.a = 1.0;

}

-- FSVel

out vec4 FragColor;

uniform sampler3D Density;
uniform sampler3D LightCache;
uniform sampler3D Velocity;

uniform vec3 LightPosition = vec3(1.0, 1.0, 2.0);
uniform vec3 LightIntensity = vec3(10.0);
uniform float Absorption = 10.0;
uniform mat4 Modelview;
uniform float FocalLength;
uniform vec2 WindowSize;
uniform vec3 RayOrigin;
uniform vec3 Ambient = vec3(0.15, 0.15, 0.20);
uniform float StepSize;
uniform int ViewSamples;
uniform vec3 velocityColor;

const bool Jitter = false;

float GetDensity(vec3 pos)
{

    return texture(Density, pos).x;
}

float GetVelocity(vec3 pos)
{
    vec3 colorHere = texture(Velocity, pos).xyz;

    if(colorHere.x > 0 || colorHere.y > 0 || colorHere.z > 0)
        return 1.0 * sqrt(colorHere.x*colorHere.x + colorHere.y*colorHere.y + colorHere.z*colorHere.z);
    else
        return 0.0;
}
/*
float GetObstacle(vec3 pos)
{
    //if ((texture(Obstacles, pos).x > 0 || texture(Obstacles, pos).y > 0 || texture(Obstacles, pos).z > 0) &&
    //    pos.x > 1.0/40.0 && pos.y > 1.0/40.0 && pos.z > 1.0/40.0 && pos.x<1.0-1.0/40.0 && pos.y<1.0-1.0/40.0 && pos.z<1.0-1.0/40.0)
    if(texture(Obstacles, pos).x > 0)
        return 200.0;
    else
        return 0.0;
}*/

struct Ray {
    vec3 Origin;
    vec3 Dir;
};

struct AABB {
    vec3 Min;
    vec3 Max;
};

bool IntersectBox(Ray r, AABB aabb, out float t0, out float t1)
{
    vec3 invR = 1.0 / r.Dir;
    vec3 tbot = invR * (aabb.Min-r.Origin);
    vec3 ttop = invR * (aabb.Max-r.Origin);
    vec3 tmin = min(ttop, tbot);
    vec3 tmax = max(ttop, tbot);
    vec2 t = max(tmin.xx, tmin.yz);
    t0 = max(t.x, t.y);
    t = min(tmax.xx, tmax.yz);
    t1 = min(t.x, t.y);
    return t0 <= t1;
}

float randhash(uint seed, float b)
{
    const float InverseMaxInt = 1.0 / 4294967295.0;
    uint i=(seed^12345391u)*2654435769u;
    i^=(i<<6u)^(i>>26u);
    i*=2654435769u;
    i+=(i<<5u)^(i>>12u);
    return float(b * i) * InverseMaxInt;
}

//this only prints the magnitude of the velocities; needs to be modified for RGB vector display
void main()
{
    //vec3 velocityColor;
    vec3 rayDirection;
    bool velocityFound = false;
    rayDirection.xy = 2.0 * gl_FragCoord.xy / WindowSize - 1.0;
    rayDirection.x /= WindowSize.y / WindowSize.x;
    rayDirection.z = -FocalLength;
    rayDirection = (vec4(rayDirection, 0) * Modelview).xyz;

    Ray eye = Ray( RayOrigin, normalize(rayDirection) );
    AABB aabb = AABB(vec3(-1), vec3(1));

    float tnear, tfar;
    IntersectBox(eye, aabb, tnear, tfar);
    if (tnear < 0.0) tnear = 0.0;

    vec3 rayStart = eye.Origin + eye.Dir * tnear;
    vec3 rayStop = eye.Origin + eye.Dir * tfar;
    rayStart = 0.5 * (rayStart + 1.0);
    rayStop = 0.5 * (rayStop + 1.0);

    vec3 pos = rayStart;
    vec3 viewDir = normalize(rayStop-rayStart) * StepSize;
    float T = 1.0;
    vec3 Lo = Ambient;

    if (Jitter) {
        uint seed = uint(gl_FragCoord.x) * uint(gl_FragCoord.y);
        pos += viewDir * (-0.5 + randhash(seed, 1.0));
    }

    float remainingLength = distance(rayStop, rayStart);

    //this goes front to back, not back to front
    for (int i=0; i < ViewSamples && remainingLength > 0.0;
        ++i, pos += viewDir, remainingLength -= StepSize) {

        float density = GetDensity(pos);
        float velocity = GetVelocity(pos); //Getvelocity should also check for closeness to boundaries, and ignore edge walls
        vec3 lightColor = vec3(1);
        if (pos.z < 0.1) {
            density = 10;
            lightColor = 3*Ambient;
        } else if (density <= 0.01 && velocity==0.0) {
            //Lo += velocity * velocityColor;
            continue;
        }

        if(velocity > 0.0){
            //T = 1.0;
            //Lo = velocityColor;
            velocityFound = true;
            break;
        }
        else
            T *= 1.0 - density * StepSize * Absorption;
        //if (T <= 0.01)
        //    break;

        vec3 Li = lightColor * texture(LightCache, pos).xxx;
        Lo += Li * T * density * StepSize;
    }

    //Lo = 1-Lo;
    if(velocityFound){
        FragColor.rgb = Lo*(1-T) + T*velocityColor;
        FragColor.a = 0.1;}
    else{

    FragColor.rgb = Lo;
    FragColor.a = 1-T;}
}
